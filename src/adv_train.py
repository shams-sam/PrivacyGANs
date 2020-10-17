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
    time_stp, logger, sep, weights_init, eigan_loss
from pix2pix import define_G, define_D, GANLoss
from resnet import get_resnet
from torchsummary import summary


def main(
        expt,
        model_name,
        device,
        gpu_id,
        optimizer,
        num_layers,
        n_classes,
        img_size,
        batch_size,
        test_batch_size,
        subset,
        init_w,
        load_w,
        ckpt_g,
        ckpt_clfs,
        n_epochs,
        lr_g,
        lr_clfs,
        ei_array,
        weight_decays,
        milestones,
        save_ckpts,
        gamma,
):
    device = torch_device(device, gpu_id[0])
    num_clfs = len([_ for _ in n_classes if _ > 0])
    Net = get_resnet(num_layers)

    net_G = define_G(cfg.num_channels[expt],
                     cfg.num_channels[expt],
                     64, gpu_id=device)
    clfs = [Net(num_channels=cfg.num_channels[expt],
                num_classes=_).to(device) for _ in n_classes if _ > 0]
    
    if len(gpu_id) > 1:
        net_G = nn.DataParallel(net_G, device_ids=gpu_id)
        clfs = [nn.DataParallel(clf, device_ids=gpu_id) for clf in clfs]

    assert len(clfs) == num_clfs

    if load_w:
        print("Loading weights...\n{}".format(ckpt_g))
        net_G.load_state_dict(torch.load(ckpt_g))
        for clf, ckpt in zip(clfs, ckpt_clfs):
            print(ckpt)
            clf.load_state_dict(torch.load(ckpt))
    elif init_w:
        print("Init weights...")
        net_G.apply(weights_init)
        for clf in clfs:
            clf.apply(weights_init)

    if optimizer == 'sgd':
        optim = torch.optim.SGD
    elif optimizer == 'adam':
        optim = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.MultiStepLR
    opt_G = torch.optim.Adam(net_G.parameters(), lr=lr_g, weight_decay=weight_decays[0])
    opt_clfs = [optim(clf.parameters(), lr=lr, weight_decay=weight_decays[1])
                for lr, clf in zip(lr_clfs, clfs)]
    sch_clfs = [scheduler(optim, milestones, gamma=gamma)
                for optim in opt_clfs]

    assert len(opt_clfs) == num_clfs

    criterionGAN = eigan_loss
    criterionNLL = nn.CrossEntropyLoss().to(device)

    train_loader = get_loader(expt, batch_size, True, img_size=img_size, subset=subset)
    valid_loader = get_loader(expt, test_batch_size, False, img_size=img_size, subset=subset)

    template = '{}'.format(model_name)

    loss_history = defaultdict(list)
    acc_history = defaultdict(list)
    for epoch in range(n_epochs):
        logging.info("Train Epoch \t Loss_G " +
                     ' '.join(["\t Clf: {}".format(_)
                               for _ in range(num_clfs)]))

        for iteration, (image, labels) in enumerate(train_loader, 1):
            real = image.to(device)
            ys = [_.to(device) for _, num_c in zip(labels, n_classes) if num_c > 0]

            with torch.no_grad():
                X = net_G(real)

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

            X = net_G(real)
            ys_hat = [clf(X) for clf in clfs]
            loss = [criterionNLL(y_hat, y)
                    for y_hat, y in zip(ys_hat,
                                        ys)]

            opt_G.zero_grad()
            loss_g = eigan_loss(loss, ei_array)
            loss_g.backward()
            opt_G.step()


            logging.info(
                '[{}]({}/{}) \t {:.4f} '.format(
                    epoch, iteration, len(train_loader),
                    loss_g.item()
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        loss_history['train_epoch'].append(epoch)
        loss_history['train_G'].append(loss_g.item())
        acc_history['train_epoch'].append(epoch)
        for idx, (l, a) in enumerate(zip(iloss, acc)):
            loss_history['train_M_{}'.format(idx)].append(l)
            acc_history['train_M_{}'.format(idx)].append(a)

        logging.info("Valid Epoch \t Loss_G " +
                     ' '.join(["\t Clf: {}".format(_) for _ in range(num_clfs)]))

        loss_g_batch = 0
        loss_m_batch = [0 for _ in range(num_clfs)]
        acc_m_batch = [0 for _ in range(num_clfs)]
        for iteration, (image, labels) in enumerate(valid_loader, 1):

            real = image.to(device)
            fake = net_G(real)
            ys = [_.to(device) for _, num_c in zip(labels, n_classes) if num_c > 0]

            ys_hat = [clf(fake) for clf in clfs]
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

            real = image.to(device)
            fake = net_G(real)

            loss_g = eigan_loss(iloss, ei_array)
            loss_g_batch += loss_g

            logging.info(
                '[{}]({}/{}) \t {:.4f} '.format(
                    epoch, iteration, len(valid_loader),
                    loss_g
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        num_samples = len(valid_loader)
        logging.info(
            '[{}](batch) \t {:.4f} '.format(
                epoch,
                loss_g_batch / num_samples
            ) +
            ' '.join(['\t {:.4f} ({:.2f})'.format(
                l/num_samples, a/num_samples) for l, a in zip(loss_m_batch, acc_m_batch)])
        )

        loss_history['valid_epoch'].append(epoch)
        loss_history['valid_G'].append(loss_g_batch/num_samples)
        acc_history['valid_epoch'].append(epoch)
        for idx, (l, a) in enumerate(zip(loss_m_batch, acc_m_batch)):
            loss_history['valid_M_{}'.format(idx)].append(l/num_samples)
            acc_history['valid_M_{}'.format(idx)].append(a/num_samples)

        for i in range(image.shape[0]):
            j = np.random.randint(0, image.shape[0])
            sample = image[j]
            label = [str(int(_[j])) for _ in labels]
            ax = plt.subplot(2, 4, i + 1)
            ax.axis('off')
            sample = sample.permute(1, 2, 0)
            plt.imshow(sample.squeeze().numpy())
            plt.savefig('{}/{}/validation/tmp.jpg'.format(cfg.ckpt_folder, expt))
            ax = plt.subplot(2, 4, 5+i)
            ax.axis('off')
            ax.set_title(" ".join(label))
            sample_G = net_G(sample.clone().permute(2, 0, 1).unsqueeze_(0).to(device))
            sample_G = sample_G.cpu().detach().squeeze()
            if sample_G.shape[0] == 3:
                sample_G = sample_G.permute(1, 2, 0)
            plt.imshow(sample_G.numpy())

            if i == 3:
                validation_plt = '{}/{}/validation/{}_{}.jpg'.format(
                    cfg.ckpt_folder, expt, model_name, epoch)
                print('Saving: {}'.format(validation_plt))
                plt.tight_layout()
                plt.savefig(validation_plt)
                break

        if epoch in save_ckpts:
            model_ckpt = '{}/{}/models/{}_g_{}.stop'.format(
                cfg.ckpt_folder, expt, model_name, epoch)
            logging.info('Model: {}'.format(model_ckpt))
            torch.save(net_G.state_dict(), model_ckpt)
            

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

    model_ckpt = '{}/{}/models/{}_g.stop'.format(
        cfg.ckpt_folder, expt, model_name)
    logging.info('Model: {}'.format(model_ckpt))
    torch.save(net_G.state_dict(), model_ckpt)
    
    for idx, clf in enumerate(clfs):
        model_ckpt = '{}/{}/models/{}_clf_{}.stop'.format(
            cfg.ckpt_folder, expt, model_name, idx)
        logging.info('Model: {}'.format(model_ckpt))
        torch.save(clf.state_dict(), model_ckpt)


if __name__ == '__main__':
    model = 'adv_train'
    args = parse()
    model += '_resnet{}_ei_{}'.format(args.num_layers, len(args.lr_clfs))
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
        num_layers=args.num_layers,
        n_classes=args.n_classes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        subset=args.subset,
        init_w=args.init_w,
        load_w=args.load_w,
        ckpt_g=args.ckpt_g,
        ckpt_clfs=args.ckpt_clfs,
        n_epochs=args.n_epochs,
        lr_g=args.lr_g,
        lr_clfs=args.lr_clfs,
        ei_array=args.ei_array,
        weight_decays=args.weight_decays,
        milestones=args.milestones,
        save_ckpts=args.save_ckpts,
        gamma=args.gamma,
    )
