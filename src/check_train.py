from collections import defaultdict
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn

from argparser import parse
import config as cfg
from data import get_loader
from utils import log_time, torch_device,\
    time_stp, logger, sep, weights_init
from pix2pix import define_G, define_D, GANLoss
from resnet import get_resnet
from utils import get_network
from torchsummary import summary


def main(
        expt,
        model_name,
        device,
        gpu_id,
        optimizer,
        arch,
        num_layers,
        n_classes,
        img_size,
        batch_size,
        test_batch_size,
        subset,
        init_w,
        ckpt_g,
        n_epochs,
        lr_clfs,
        weight_decays,
        milestones,
        gamma,
):
    device = torch_device(device, gpu_id[0])
    num_clfs = len([_ for _ in n_classes if _ > 0])
    if arch == 'resnet':
        print('Using resnet')
        Net = get_resnet(num_layers)
    else:
        print('Using {}'.format(arch))
        Net = get_network(arch, num_layers)

    net_G = define_G(cfg.num_channels[expt],
                     cfg.num_channels[expt],
                     64, gpu_id=device)
    clfs = [Net(num_channels=cfg.num_channels[expt],
                num_classes=_).to(device) for _ in n_classes if _ > 0]

    if len(gpu_id) > 1:
        net_G = nn.DataParallel(net_G, device_ids=gpu_id)
        clfs = [nn.DataParallel(clf, device_ids=gpu_id) for clf in clfs]
    
    assert len(clfs) == num_clfs

    print("Loading weights...\n{}".format(ckpt_g))
    net_G.load_state_dict(torch.load(ckpt_g))
    if init_w:
        print("Init weights...")
        for clf in clfs:
            clf.apply(weights_init)

    scheduler = torch.optim.lr_scheduler.MultiStepLR
    if optimizer == 'sgd':
        opt_clfs = [torch.optim.SGD(clf.parameters(), lr=lr,  momentum=0.9,
                                    weight_decay=weight_decays[0])
                    for lr, clf in zip(lr_clfs, clfs)]
    elif optimizer == 'adam':
        opt_clfs = [torch.optim.SGD(clf.parameters(), lr=lr,  weight_decay=weight_decays[0])
                                    for lr, clf in zip(lr_clfs, clfs)]
    sch_clfs = [scheduler(optim, milestones, gamma=gamma)
                for optim in opt_clfs]

    assert len(opt_clfs) == num_clfs

    criterionNLL = nn.CrossEntropyLoss().to(device)

    train_loader = get_loader(expt, batch_size, True, img_size=img_size, subset=subset)
    valid_loader = get_loader(expt, test_batch_size, False, img_size=img_size, subset=subset)

    template = '{}'.format(model_name)

    loss_history = defaultdict(list)
    acc_history = defaultdict(list)
    for epoch in range(n_epochs):
        logging.info("Train Epoch " + ' '.join(["\t Clf: {}".format(_)
                               for _ in range(num_clfs)]))

        for iteration, (image, labels) in enumerate(train_loader, 1):
            real = image.to(device)

            with torch.no_grad():
                X = net_G(real)
            ys = [_.to(device) for _ in labels]

            [opt.zero_grad() for opt in opt_clfs]
            ys_hat = [clf(X) for clf in clfs]
            loss = [criterionNLL(y_hat, y)
                    for y_hat, y in zip(ys_hat,
                                        ys)]
            ys_hat = [_.argmax(1, keepdim=True)
                      for _ in ys_hat]
            acc = [y_hat.eq(y.view_as(y_hat)).sum().item()/len(y)
                   for y_hat, y in zip(ys_hat, ys)]
            [l.backward() for l in loss]
            [opt.step() for opt in opt_clfs]

            iloss = [l.item() for l in loss]
            assert len(iloss) == num_clfs

            logging.info(
                '[{}]({}/{}) '.format(
                    epoch, iteration, len(train_loader),
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        loss_history['train_epoch'].append(epoch)
        acc_history['train_epoch'].append(epoch)
        for idx, (l, a) in enumerate(zip(iloss, acc)):
            loss_history['train_M_{}'.format(idx)].append(l)
            acc_history['train_M_{}'.format(idx)].append(a)

        logging.info("Valid Epoch " +
                     ' '.join(["\t Clf: {}".format(_) for _ in range(num_clfs)]))

        loss_m_batch = [0 for _ in range(num_clfs)]
        acc_m_batch = [0 for _ in range(num_clfs)]
        for iteration, (image, labels) in enumerate(valid_loader, 1):

            X = net_G(image.to(device))
            ys = [_.to(device) for _ in labels]

            ys_hat = [clf(X) for clf in clfs]
            loss = [criterionNLL(y_hat, y)
                    for y_hat, y in zip(ys_hat, ys)]
            ys_hat = [_.argmax(1, keepdim=True)
                      for _ in ys_hat]
            acc = [y_hat.eq(y.view_as(y_hat)).sum().item()/len(y)
                   for y_hat, y in zip(ys_hat, ys)]

            iloss = [l.item() for l in loss]
            for idx, (l, a) in enumerate(zip(iloss, acc)):
                loss_m_batch[idx] += l
                acc_m_batch[idx] += a

            logging.info(
                '[{}]({}/{}) '.format(
                    epoch, iteration, len(valid_loader),
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        num_samples = len(valid_loader)
        logging.info(
            '[{}](batch) '.format(
                epoch,
            ) +
            ' '.join(['\t {:.4f} ({:.2f})'.format(
                l/num_samples, a/num_samples) for l, a in zip(loss_m_batch, acc_m_batch)])
        )

        num_samples = len(valid_loader)
        loss_history['valid_epoch'].append(epoch)
        acc_history['valid_epoch'].append(epoch)
        for idx, (l, a) in enumerate(zip(loss_m_batch, acc_m_batch)):
            loss_history['valid_M_{}'.format(idx)].append(l/num_samples)
            acc_history['valid_M_{}'.format(idx)].append(a/num_samples)

        [sch.step() for sch in sch_clfs]

    train_loss_keys = [
        _ for _ in loss_history if 'train' in _ and 'epoch' not in _]
    valid_loss_keys = [
        _ for _ in loss_history if 'valid' in _ and 'epoch' not in _]
    train_acc_keys = [
        _ for _ in acc_history if 'train' in _ and 'epoch' not in _]
    valid_acc_keys = [
        _ for _ in acc_history if 'valid' in _ and 'epoch' not in _]

    cols = 5
    rows = len(train_loss_keys)//cols + 1
    fig = plt.figure(figsize=(7*cols, 5*rows))
    base = cols*100 + rows*10
    for idx, (tr_l, val_l) in enumerate(zip(train_loss_keys, valid_loss_keys)):
        ax = fig.add_subplot(rows, cols, idx+1)
        ax.plot(loss_history['train_epoch'], loss_history[tr_l], 'b.:')
        ax.plot(loss_history['valid_epoch'], loss_history[val_l], 'bs-.')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title(tr_l[6:])
        ax.grid()
        if tr_l in acc_history:
            ax2 = plt.twinx()
            ax2.plot(acc_history['train_epoch'], acc_history[tr_l], 'r.:')
            ax2.plot(acc_history['valid_epoch'], acc_history[val_l], 'rs-.')
            ax2.set_ylabel('accuracy')
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    plt_ckpt = '{}/{}/plots/{}.jpg'.format(
        cfg.ckpt_folder, expt, model_name)
    logging.info('Plot: {}'.format(plt_ckpt))
    plt.savefig(plt_ckpt, bbox_inches='tight', dpi=80)

    hist_ckpt = '{}/{}/history/{}.pkl'.format(
        cfg.ckpt_folder, expt, model_name)
    logging.info('History: {}'.format(hist_ckpt))
    pkl.dump((loss_history, acc_history), open(hist_ckpt, 'wb'))
    
    for idx, clf in enumerate(clfs):
        model_ckpt = '{}/{}/models/{}_clf_{}.stop'.format(
            cfg.ckpt_folder, expt, model_name, idx)
        logging.info('Model: {}'.format(model_ckpt))
        torch.save(clf.state_dict(), model_ckpt)


if __name__ == '__main__':
    model = 'check_train'
    args = parse()
    model = '{}_{}{}_optim_{}_ei_{}_epochs_{}'.format(
        model, args.arch, args.num_layers, args.optimizer, len(args.lr_clfs), args.n_epochs)
    pr_time, fl_time = time_stp()
    logger(args.expt, model)

    log_time('Start', pr_time)
    sep()
    logging.info(json.dumps(args.__dict__, indent=2))

    main(
        expt=args.expt,
        model_name=model,
        device=args.device,
        gpu_id=args.gpu_id,
        optimizer=args.optimizer,
        arch=args.arch,
        num_layers=args.num_layers,
        n_classes=args.n_classes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        subset=args.subset,
        init_w=args.init_w,
        ckpt_g=args.ckpt_g,
        n_epochs=args.n_epochs,
        lr_clfs=args.lr_clfs,
        weight_decays=args.weight_decays,
        milestones=args.milestones,
        gamma=args.gamma,
    )
