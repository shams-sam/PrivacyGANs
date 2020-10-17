import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")  # noqa

from models.eigan import GeneratorFCN, DiscriminatorFCN
from preprocessing import get_data
from common.torchsummary import summary
from common.utility import log_shapes, log_time, torch_device,\
    time_stp, load_processed_data, logger, sep, to_categorical, weights_init
import common.config as cfg
from common.data import get_loader
from common.argparser import eigan_argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import torch.nn as nn
import torch.utils.data as utils
from tqdm import tqdm
from models.resnet import resnet18 as Net
from models.pix2pix import define_G


def main(
        model,
        time_stamp,
        device,
        ally_classes,
        advr_classes,
        batch_size,
        n_epochs,
        shuffle,
        init_weight,
        lr_encd,
        lr_ally,
        lr_advr,
        alpha,
        expt,
        encoder_ckpt,
        ally_ckpts,
        advr_ckpts,
        marker
):

    device = torch_device(device=device)

    encoder = define_G(cfg.num_channels[expt],
                       cfg.num_channels[expt],
                       64, gpu_id=device)
    encoder.load_state_dict(torch.load(encoder_ckpt))
    sep()
    logging.info("Loaded: {}".format(encoder_ckpt))
    allies = [Net(num_classes=_).to(device) for _ in ally_classes]
    advrs = [Net(num_classes=_).to(device) for _ in advr_classes]
    for ally, ckpt in zip(allies, ally_ckpts):
        logging.info("Loaded: {}".format(ckpt))
        ally.load_state_dict(torch.load(ckpt))
    for advr, ckpt in zip(advrs, advr_ckpts):
        logging.info("Loaded: {}".format(ckpt))
        advr.load_state_dict(torch.load(ckpt))
    sep()

    optim = torch.optim.Adam
    criterionNLL = nn.NLLLoss()

    optimizer_encd = optim(encoder.parameters(), lr=lr_encd)
    optimizer_ally = [optim(ally.parameters(), lr=lr)
                      for lr, ally in zip(lr_ally, allies)]
    optimizer_advr = [optim(advr.parameters(), lr=lr)
                      for lr, advr in zip(lr_advr, advrs)]

    dataloader_train = get_loader(expt, batch_size, True)
    dataloader_valid = get_loader(expt, batch_size, False)

    epochs_train = []
    epochs_valid = []
    encd_loss_train = []
    encd_loss_valid = []
    ally_loss_train = []
    ally_loss_valid = []
    advr_loss_train = []
    advr_loss_valid = []

    template = '{}_{}_{}'.format(expt, model, marker)

    log_head = '{} \t {} \t {}'.format(
        'Epoch',
        'Encd Tr',
        'Encd Val',
    )
    for _ in range(len(ally_classes)):
        log_head += ' \t {} \t {}'.format(
            'A{} tr'.format(_), 'A{} val'.format(_))
    for _ in range(len(advr_classes)):
        log_head += ' \t {} \t {}'.format(
            'V{} tr'.format(_), 'V{} val'.format(_))
    logging.info(log_head)

    encoder.train()
    for ally in allies:
        ally.train()
    for advr in advrs:
        advr.train()

    for epoch in range(n_epochs):

        nsamples = 0
        iloss = 0
        for i, data in tqdm(enumerate(dataloader_train, 0),
                            total=len(dataloader_train)):
            X_train_torch = data[0].to(device)
            y_ally_train_torch = [
                (data[1] % 2 == 0).type(torch.int64).to(device)]
            y_advr_train_torch = [
                data[1].to(device),
                #     (data[1] >= 5).type(torch.int64).to(device)
            ]

            optimizer_encd.zero_grad()
            # Forward pass
            X_train_encoded = encoder(X_train_torch)
            y_ally_train_hat_torch = [ally(X_train_encoded) for ally in allies]
            y_advr_train_hat_torch = [advr(X_train_encoded) for advr in advrs]
            # Compute Loss
            loss_ally = [criterionNLL(y_hat, y)
                         for y_hat, y in zip(y_ally_train_hat_torch,
                                             y_ally_train_torch)]
            loss_advr = [criterionNLL(y_hat, y)
                         for y_hat, y in zip(
                y_advr_train_hat_torch,
                y_advr_train_torch)]
            loss_encd = sum(loss_ally) + sum(loss_advr)
            # Backward pass
            loss_encd.backward()
            optimizer_encd.step()

            nsamples += 1
            iloss += loss_encd.item()

        epochs_train.append(epoch)
        encd_loss_train.append(iloss/nsamples)

        nsamples = 0
        iloss_ally = np.array([0] * len(allies))
        iloss_advr = np.array([0] * len(advrs))
        for i, data in tqdm(enumerate(dataloader_train, 0),
                            total=len(dataloader_train)):
            X_train_torch = data[0].to(device)
            y_ally_train_torch = [
                (data[1] % 2 == 0).type(torch.int64).to(device)]
            y_advr_train_torch = [
                data[1].to(device),
                #     (data[1] >= 5).type(torch.int64).to(device)
            ]

            [opt_ally.zero_grad() for opt_ally in optimizer_ally]
            X_train_encoded = encoder(X_train_torch)
            y_ally_train_hat_torch = [ally(X_train_encoded) for ally in allies]
            loss_ally = [criterionNLL(y_hat, y)
                         for y_hat, y in zip(y_ally_train_hat_torch,
                                             y_ally_train_torch)]
            [l_ally.backward() for l_ally in loss_ally]
            [opt_ally.step() for opt_ally in optimizer_ally]

            [opt_advr.zero_grad() for opt_advr in optimizer_advr]
            X_train_encoded = encoder(X_train_torch)
            y_advr_train_hat_torch = [advr(X_train_encoded) for advr in advrs]
            loss_advr = [criterionNLL(y_hat, y)
                         for y_hat, y in zip(y_advr_train_hat_torch,
                                             y_advr_train_torch)]
            [l_advr.backward(retain_graph=True) for l_advr in loss_advr]
            [opt_advr.step() for opt_advr in optimizer_advr]

            nsamples += 1
            iloss_ally = iloss_ally + \
                np.array([l_ally.item() for l_ally in loss_ally])
            iloss_advr = iloss_advr + \
                np.array([l_advr.item() for l_advr in loss_advr])

        ally_loss_train.append(iloss_ally/nsamples)
        advr_loss_train.append(iloss_advr/nsamples)

        if epoch % int(n_epochs/10) != 0:
            continue

        nsamples = 0
        iloss = 0
        iloss_ally = np.array([0] * len(allies))
        iloss_advr = np.array([0] * len(advrs))

        for i, data in tqdm(enumerate(dataloader_valid, 0),
                            total=len(dataloader_valid)):
            X_valid_torch = data[0].to(device)
            y_ally_valid_torch = [
                (data[1] % 2 == 0).type(torch.int64).to(device)]
            y_advr_valid_torch = [
                data[1].to(device),
                #     (data[1] >= 5).type(torch.int64).to(device)
            ]

            X_valid_encoded = encoder(X_valid_torch)
            y_ally_valid_hat_torch = [ally(X_valid_encoded) for ally in allies]
            y_advr_valid_hat_torch = [advr(X_valid_encoded) for advr in advrs]
            # Compute Loss
            loss_ally = [criterionNLL(y_hat, y)
                         for y_hat, y in zip(y_ally_valid_hat_torch,
                                             y_ally_valid_torch)]
            loss_advr = [criterionNLL(y_hat, y)
                         for y_hat, y in zip(y_advr_valid_hat_torch,
                                             y_advr_valid_torch)]
            loss_encd = sum(loss_ally) - sum(loss_advr)
            if i < 4:
                sample = X_valid_torch[0].cpu().detach().squeeze().numpy()
                ax = plt.subplot(2, 4, i+1)
                plt.imshow(sample)
                ax.axis('off')
                output = X_valid_encoded[0].cpu().detach().squeeze().numpy()
                ax = plt.subplot(2, 4, i+5)
                plt.imshow(output)
                ax.axis('off')

                if i == 3:
                    validation_plt = 'ckpts/{}/validation/{}_{}.jpg'.format(
                        expt, template, epoch)
                    print('Saving: {}'.format(validation_plt))
                    plt.savefig(validation_plt)

            nsamples += 1
            iloss += loss_encd.item()
            iloss_ally = iloss_ally + \
                np.array([l_ally.item() for l_ally in loss_ally])
            iloss_advr = iloss_advr + \
                np.array([l_advr.item() for l_advr in loss_advr])

        epochs_valid.append(epoch)
        encd_loss_valid.append(iloss/nsamples)
        ally_loss_valid.append(iloss_ally/nsamples)
        advr_loss_valid.append(iloss_advr/nsamples)

        logging.info('{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.
                     format(
                         epoch,
                         encd_loss_train[-1],
                         encd_loss_valid[-1],
                         ally_loss_train[-1][0],
                         ally_loss_valid[-1][0],
                         advr_loss_train[-1][0],
                         advr_loss_valid[-1][0],
                     ))

    ally_loss_train = np.vstack(ally_loss_train)
    ally_loss_valid = np.vstack(ally_loss_valid)
    advr_loss_train = np.vstack(advr_loss_train)
    advr_loss_valid = np.vstack(advr_loss_valid)

    fig = plt.figure(figsize=(15, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
#     ax4 = fig.add_subplot(224)
    ax1.plot(epochs_train, encd_loss_train, 'r', label='encd tr')
    ax1.plot(epochs_valid, encd_loss_valid, 'r--', label='encd val')
    ax1.legend()
    for col, c, ax in zip(range(ally_loss_train.shape[1]), ['b'], [ax2]):
        ax.plot(epochs_train, ally_loss_train[:, col],
                '{}.:'.format(c), label='ally {} tr'.format(col))
        ax.plot(epochs_valid, ally_loss_valid[:, col],
                '{}s-.'.format(c), label='ally {} val'.format(col))
        ax.legend()
    for col, c, ax in zip(range(advr_loss_train.shape[1]), ['g'], [ax3]):
        ax.plot(epochs_train, advr_loss_train[:, col],
                '{}.:'.format(c), label='advr {} tr'.format(col))
        ax.plot(epochs_valid, advr_loss_valid[:, col],
                '{}s-.'.format(c), label='advr {} val'.format(col))
        ax.legend()

    plot_location = 'ckpts/{}/plots/{}.png'.format(
        expt, template)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location)
    checkpoint_location = 'ckpts/{}/history/{}.pkl'.format(
        expt, template)
    logging.info('Saving: {}'.format(checkpoint_location))
    pkl.dump((
        epochs_train, epochs_valid,
        encd_loss_train, encd_loss_valid,
        ally_loss_train, ally_loss_valid,
        advr_loss_train, advr_loss_valid,
    ), open(checkpoint_location, 'wb'))

    model_ckpt = 'ckpts/{}/models/{}.pkl'.format(
        expt, template)
    logging.info('Saving: {}'.format(model_ckpt))
    torch.save(encoder.state_dict(), model_ckpt)


if __name__ == "__main__":
    expt = 'mnist'
    model = 'eigan'
    marker = 'A'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    args = eigan_argparse()
    main(
        model=model,
        time_stamp=fl_time,
        device=args['device'],
        ally_classes=args['n_ally'],
        advr_classes=args['n_advr'],
        batch_size=args['batch_size'],
        n_epochs=args['n_epochs'],
        shuffle=args['shuffle'] == 1,
        init_weight=args['init_w'] == 1,
        lr_encd=args['lr_encd'],
        lr_ally=args['lr_ally'],
        lr_advr=args['lr_advr'],
        alpha=args['alpha'],
        expt=args['expt'],
        encoder_ckpt=args['encd_ckpt'],
        ally_ckpts=args['ally_ckpts'],
        advr_ckpts=args['advr_ckpts'],
        marker=marker
    )
    log_time('End', time_stp()[0])
    sep()
