import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import torch.nn as nn
import torch.utils.data as utils

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")  # noqa

from common.argparser import eigan_argparse
import common.config as cfg
from common.utility import log_shapes, log_time, torch_device,\
    time_stp, load_processed_data, logger, sep, to_categorical, weights_init
from common.torchsummary import summary

from preprocessing import get_data
from common.data import get_loader

from models.pix2pix import define_G, define_D, GANLoss
from models.resnet import resnet18 as Net


def main(
        model,
        device,
        ally_classes,
        advr_classes,
        batch_size,
        n_epochs,
        lr_encd,
        lr_ally,
        lr_advr,
        expt,
        marker,
):

    device = torch_device(device=device)

    netG = define_G(cfg.num_channels[expt],
                    cfg.num_channels[expt],
                    64, gpu_id=device)
    netD = define_D(cfg.num_channels[expt],
                    64, 'basic', gpu_id=device)
    allies = [Net(num_classes=_).to(device) for _ in ally_classes]
    advrs = [Net(num_classes=_).to(device) for _ in advr_classes]

    netG.apply(weights_init)
    netD.apply(weights_init)
    for ally in allies:
        ally.apply(weights_init)
    for advr in advrs:
        advr.apply(weights_init)

    optim = torch.optim.Adam
    optG = optim(netG.parameters(), lr=lr_encd)
    optD = optim(netD.parameters(), lr=lr_encd)
    opt_ally = [optim(ally.parameters(), lr=lr)
                for lr, ally in zip(lr_ally, allies)]
    opt_advr = [optim(advr.parameters(), lr=lr)
                for lr, advr in zip(lr_advr, advrs)]

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionNLL = nn.NLLLoss()

    train_loader = get_loader(expt, batch_size, True)
    valid_loader = get_loader(expt, cfg.num_tests[expt], False)
    for data in valid_loader:
        valid_dataset = data[0]

    template = '{}_{}_{}'.format(expt, model, marker)
    for epoch in range(n_epochs):
        for iteration, (image, label) in enumerate(train_loader, 1):
            # forward
            real = image.to(device)
            fake = netG(real)

            optD.zero_grad()
            pred_fake = netD.forward(fake.detach())
            loss_d_fake = criterionGAN(pred_fake, False)
            pred_real = netD.forward(real)
            loss_d_real = criterionGAN(pred_real, True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optD.step()

            optG.zero_grad()
            pred_fake = netD.forward(fake)
            loss_g_gan = criterionGAN(pred_fake, True)
            loss_g_l1 = criterionL1(fake, real) * 10
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            optG.step()
            logging.info("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(train_loader), loss_d.item(), loss_g.item()))

            X_train_torch = image.to(device)
            y_ally_train_torch = [
                (label % 2 == 0).type(torch.int64).to(device)]
            y_advr_train_torch = [
                label.to(device),
                #     (data[1] >= 5).type(torch.int64).to(device)
            ]

            [opt.zero_grad() for opt in opt_ally]
            y_ally_train_hat_torch = [ally(X_train_torch) for ally in allies]
            loss_ally = [criterionNLL(y_hat, y)
                         for y_hat, y in zip(y_ally_train_hat_torch,
                                             y_ally_train_torch)]
            y_hats = [_.argmax(1, keepdim=True)
                      for _ in y_ally_train_hat_torch]
            acc_ally = [y_hat.eq(y.view_as(y_hat)).sum().item()/len(y)
                        for y_hat, y in zip(y_hats, y_ally_train_torch)]
            [l_ally.backward() for l_ally in loss_ally]
            [opt.step() for opt in opt_ally]

            [opt.zero_grad() for opt in opt_advr]
            y_advr_train_hat_torch = [advr(X_train_torch) for advr in advrs]
            loss_advr = [criterionNLL(y_hat, y)
                         for y_hat, y in zip(y_advr_train_hat_torch,
                                             y_advr_train_torch)]
            y_hats = [_.argmax(1, keepdim=True)
                      for _ in y_advr_train_hat_torch]
            acc_advr = [y_hat.eq(y.view_as(y_hat)).sum().item()/len(y)
                        for y_hat, y in zip(y_hats, y_advr_train_torch)]
            [l_advr.backward(retain_graph=True) for l_advr in loss_advr]
            [opt.step() for opt in opt_advr]

            iloss_ally = np.array([l_ally.item() for l_ally in loss_ally])
            iloss_advr = np.array([l_advr.item() for l_advr in loss_advr])

            logging.info('\t {:.4f} ({:.4f}) \t {:.4f} ({:.4f})'.format(
                iloss_ally.item(), acc_ally[0], iloss_advr.item(), acc_advr[0]))

        for i in range(len(valid_dataset)):
            j = np.random.randint(0, len(valid_dataset))
            sample = valid_dataset[j]
            ax = plt.subplot(2, 4, i + 1)
            plt.tight_layout()
            ax.axis('off')
            plt.imshow(sample.squeeze().numpy())

            sample = netG(sample.unsqueeze_(0).to(device))
            ax = plt.subplot(2, 4, 5+i)
            plt.tight_layout()
            ax.axis('off')
            plt.imshow(sample.cpu().detach().squeeze().numpy())

            if i == 3:
                validation_plt = 'ckpts/{}/validation/{}_{}.jpg'.format(
                    expt, template, epoch)
                print('Saving: {}'.format(validation_plt))
                plt.savefig(validation_plt)
                break

    model_ckpt = 'ckpts/{}/models/{}_encoder.stop'.format(expt, template)
    logging.info('Saving: {}'.format(model_ckpt))
    torch.save(netG.state_dict(), model_ckpt)

    for idx, ally in enumerate(allies):
        model_ckpt = 'ckpts/{}/models/{}_ally_{}.stop'.format(
            expt, template, idx)
        logging.info('Saving: {}'.format(model_ckpt))
        torch.save(ally.state_dict(), model_ckpt)

    for idx, advr in enumerate(advrs):
        model_ckpt = 'ckpts/{}/models/{}_advr_{}.stop'.format(
            expt, template, idx)
        logging.info('Saving: {}'.format(model_ckpt))
        torch.save(advr.state_dict(), model_ckpt)


if __name__ == '__main__':
    expt = 'mnist'
    model = 'encd_pretrain'
    marker = 'A'

    pr_time, fl_time = time_stp()
    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    args = eigan_argparse()
    main(
        model=model,
        device=args['device'],
        ally_classes=args['n_ally'],
        advr_classes=args['n_advr'],
        batch_size=args['batch_size'],
        n_epochs=args['n_epochs'],
        lr_encd=args['lr_encd'],
        lr_ally=args['lr_ally'],
        lr_advr=args['lr_advr'],
        expt=args['expt'],
        marker=marker,
    )
