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
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.argparser import eigan_argparse
from common.utility import log_shapes, log_time, torch_device,\
    time_stp, load_processed_data, logger, sep, to_categorical, weights_init
from common.torchsummary import summary

from preprocessing import get_data

from models.eigan import GeneratorFCN, DiscriminatorFCN


def main(
        model,
        time_stamp,
        device,
        ngpu,
        ally_classes,
        advr_1_classes,
        advr_2_classes,
        encoding_dim,
        hidden_dim,
        leaky,
        activation,
        test_size,
        batch_size,
        n_epochs,
        shuffle,
        init_weight,
        lr_encd,
        lr_ally,
        lr_advr_1,
        lr_advr_2,
        alpha,
        g_reps,
        d_reps,
        expt,
        marker
        ):

    device = torch_device(device=device)

    X_normalized_train, X_normalized_valid,\
        y_ally_train, y_ally_valid, \
        y_advr_1_train, y_advr_1_valid, \
        y_advr_2_train, y_advr_2_valid = get_data(expt, test_size)

    encoder = GeneratorFCN(
        X_normalized_train.shape[1], hidden_dim, encoding_dim,
        leaky, activation).to(device)
    ally = DiscriminatorFCN(
        encoding_dim, hidden_dim, ally_classes,
        leaky).to(device)
    advr_1 = DiscriminatorFCN(
        encoding_dim, hidden_dim, advr_1_classes,
        leaky).to(device)
    advr_2 = DiscriminatorFCN(
        encoding_dim, hidden_dim, advr_2_classes,
        leaky).to(device)

    if init_weight:
        sep()
        logging.info('applying weights_init ...')
        encoder.apply(weights_init)
        ally.apply(weights_init)
        advr_1.apply(weights_init)
        advr_2.apply(weights_init)

    sep('encoder')
    summary(encoder, input_size=(1, X_normalized_train.shape[1]))
    sep('ally')
    summary(ally, input_size=(1, encoding_dim))
    sep('advr_1')
    summary(advr_1, input_size=(1, encoding_dim))
    sep('advr_2')
    summary(advr_2, input_size=(1, encoding_dim))

    optim = torch.optim.Adam
    criterionBCEWithLogits = nn.BCEWithLogitsLoss()
    criterionCrossEntropy = nn.CrossEntropyLoss()

    optimizer_encd = optim(
        encoder.parameters(),
        lr=lr_encd,
        weight_decay=lr_encd
    )
    optimizer_ally = optim(
        ally.parameters(),
        lr=lr_ally,
        weight_decay=lr_ally
    )
    optimizer_advr_1 = optim(
        advr_1.parameters(),
        lr=lr_advr_1,
        weight_decay=lr_advr_1
    )
    optimizer_advr_2 = optim(
        advr_2.parameters(),
        lr=lr_advr_2,
        weight_decay=lr_advr_2
    )

    dataset_train = utils.TensorDataset(
        torch.Tensor(X_normalized_train),
        torch.Tensor(y_ally_train),
        torch.Tensor(y_advr_1_train),
        torch.Tensor(y_advr_2_train),
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1
    )

    dataset_valid = utils.TensorDataset(
        torch.Tensor(X_normalized_valid),
        torch.Tensor(y_ally_valid),
        torch.Tensor(y_advr_1_valid),
        torch.Tensor(y_advr_2_valid)
    )

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1
    )

    epochs_train = []
    epochs_valid = []
    encd_loss_train = []
    encd_loss_valid = []
    ally_loss_train = []
    ally_loss_valid = []
    advr_1_loss_train = []
    advr_1_loss_valid = []
    advr_2_loss_train = []
    advr_2_loss_valid = []

    logging.info('{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {}'.format(
                'Epoch',
                'Encd Train',
                'Encd Valid',
                'Ally Train',
                'Ally Valid',
                'Advr 1 Train',
                'Advr 1 Valid',
                'Advr 2 Train',
                'Advr 2 Valid',
                ))

    for epoch in range(n_epochs):

        encoder.train()
        ally.eval()
        advr_1.eval()
        advr_1.eval()

        for __ in range(g_reps):
            nsamples = 0
            iloss = 0
            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = data[0].to(device)
                y_ally_train_torch = data[1].to(device)
                y_advr_1_train_torch = data[2].to(device)
                y_advr_2_train_torch = data[3].to(device)

                optimizer_encd.zero_grad()
                # Forward pass
                X_train_encoded = encoder(X_train_torch)
                y_ally_train_hat_torch = ally(X_train_encoded)
                y_advr_1_train_hat_torch = advr_1(X_train_encoded)
                y_advr_2_train_hat_torch = advr_2(X_train_encoded)
                # Compute Loss
                loss_ally = criterionBCEWithLogits(
                    y_ally_train_hat_torch, y_ally_train_torch)
                loss_advr_1 = criterionCrossEntropy(
                    y_advr_1_train_hat_torch,
                    torch.argmax(y_advr_1_train_torch, 1))
                loss_advr_2 = criterionBCEWithLogits(
                    y_advr_2_train_hat_torch,
                    y_advr_2_train_torch)
                loss_encd = - loss_ally + loss_advr_1 - loss_advr_2
                # Backward pass
                loss_encd.backward()
                optimizer_encd.step()

                nsamples += 1
                iloss += loss_encd.item()

        epochs_train.append(epoch)
        encd_loss_train.append(iloss/nsamples)

        encoder.eval()
        ally.train()
        advr_1.train()
        advr_2.train()

        for __ in range(d_reps):
            nsamples = 0
            iloss_ally = 0
            iloss_advr_1 = 0
            iloss_advr_2 = 0
            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = data[0].to(device)
                y_ally_train_torch = data[1].to(device)
                y_advr_1_train_torch = data[2].to(device)
                y_advr_2_train_torch = data[3].to(device)

                optimizer_ally.zero_grad()
                X_train_encoded = encoder(X_train_torch)
                y_ally_train_hat_torch = ally(X_train_encoded)
                loss_ally = criterionBCEWithLogits(
                    y_ally_train_hat_torch, y_ally_train_torch)
                loss_ally.backward()
                optimizer_ally.step()

                optimizer_advr_1.zero_grad()
                X_train_encoded = encoder(X_train_torch)
                y_advr_1_train_hat_torch = advr_1(X_train_encoded)
                loss_advr_1 = criterionCrossEntropy(
                    y_advr_1_train_hat_torch,
                    torch.argmax(y_advr_1_train_torch, 1))
                loss_advr_1.backward()
                optimizer_advr_1.step()

                optimizer_advr_2.zero_grad()
                X_train_encoded = encoder(X_train_torch)
                y_advr_2_train_hat_torch = advr_2(X_train_encoded)
                loss_advr_2 = criterionBCEWithLogits(
                    y_advr_2_train_hat_torch,
                    y_advr_2_train_torch)
                loss_advr_2.backward()
                optimizer_advr_2.step()

                nsamples += 1
                iloss_ally += loss_ally.item()
                iloss_advr_1 += loss_advr_1.item()
                iloss_advr_2 += loss_advr_2.item()

        ally_loss_train.append(iloss_ally/nsamples)
        advr_1_loss_train.append(iloss_advr_1/nsamples)
        advr_2_loss_train.append(iloss_advr_2/nsamples)

        if epoch % int(n_epochs/10) != 0:
            continue

        encoder.eval()
        ally.eval()
        advr_1.eval()
        advr_2.eval()

        nsamples = 0
        iloss = 0
        iloss_ally = 0
        iloss_advr_1 = 0
        iloss_advr_2 = 0

        for i, data in enumerate(dataloader_valid, 0):
            X_valid_torch = data[0].to(device)
            y_ally_valid_torch = data[1].to(device)
            y_advr_1_valid_torch = data[2].to(device)
            y_advr_2_valid_torch = data[3].to(device)

            X_valid_encoded = encoder(X_valid_torch)
            y_ally_valid_hat_torch = ally(X_valid_encoded)
            y_advr_1_valid_hat_torch = advr_1(X_valid_encoded)
            y_advr_2_valid_hat_torch = advr_2(X_valid_encoded)

            valid_loss_ally = criterionBCEWithLogits(
                y_ally_valid_hat_torch, y_ally_valid_torch)
            valid_loss_advr_1 = criterionCrossEntropy(
                y_advr_1_valid_hat_torch,
                torch.argmax(y_advr_1_valid_torch, 1))
            valid_loss_advr_2 = criterionBCEWithLogits(
                y_advr_2_valid_hat_torch, y_advr_2_valid_torch)
            valid_loss_encd = - valid_loss_ally + \
                valid_loss_advr_1 - valid_loss_advr_2

            nsamples += 1
            iloss += valid_loss_encd.item()
            iloss_ally += valid_loss_ally.item()
            iloss_advr_1 += valid_loss_advr_1.item()
            iloss_advr_2 += valid_loss_advr_2.item()

        epochs_valid.append(epoch)
        encd_loss_valid.append(iloss/nsamples)
        ally_loss_valid.append(iloss_ally/nsamples)
        advr_1_loss_valid.append(iloss_advr_1/nsamples)
        advr_2_loss_valid.append(iloss_advr_2/nsamples)

        logging.info(
            '{} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f}'.
            format(
                epoch,
                encd_loss_train[-1],
                encd_loss_valid[-1],
                ally_loss_train[-1],
                ally_loss_valid[-1],
                advr_1_loss_train[-1],
                advr_1_loss_valid[-1],
                advr_2_loss_train[-1],
                advr_2_loss_valid[-1],
            ))

    config_summary = '{}_device_{}_dim_{}_hidden_{}_batch_{}_epochs_{}_lrencd_{}_lrally_{}_lradvr_{}_tr_{:.4f}_val_{:.4f}'\
        .format(
            marker,
            device,
            encoding_dim,
            hidden_dim,
            batch_size,
            n_epochs,
            lr_encd,
            lr_ally,
            lr_advr_1,
            encd_loss_train[-1],
            advr_1_loss_valid[-1],
        )

    plt.plot(epochs_train, encd_loss_train, 'r')
    plt.plot(epochs_valid, encd_loss_valid, 'r--')
    plt.plot(epochs_train, ally_loss_train, 'b')
    plt.plot(epochs_valid, ally_loss_valid, 'b--')
    plt.plot(epochs_train, advr_1_loss_train, 'g')
    plt.plot(epochs_valid, advr_1_loss_valid, 'g--')
    plt.plot(epochs_train, advr_2_loss_train, 'y')
    plt.plot(epochs_valid, advr_2_loss_valid, 'y--')
    plt.legend([
        'encoder train', 'encoder valid',
        'ally train', 'ally valid',
        'advr 1 train', 'advr 1 valid',
        'advr 2 train', 'advr 2 valid'
    ])
    plt.title("{} on {} training {}".format(model, expt, config_summary))

    plot_location = 'plots/{}/{}_training_{}_{}.png'.format(
        expt, model, time_stamp, config_summary)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location)
    checkpoint_location = \
        'checkpoints/{}/{}_training_history_{}_{}.pkl'.format(
            expt, model, time_stamp, config_summary)
    logging.info('Saving: {}'.format(checkpoint_location))
    pkl.dump((
        epochs_train, epochs_valid,
        encd_loss_train, encd_loss_valid,
        ally_loss_train, ally_loss_valid,
        advr_1_loss_train, advr_1_loss_valid,
        advr_2_loss_train, advr_2_loss_valid,
    ), open(checkpoint_location, 'wb'))

    model_ckpt = 'checkpoints/{}/{}_torch_model_{}_{}.pkl'.format(
            expt, model, time_stamp, config_summary)
    logging.info('Saving: {}'.format(model_ckpt))
    torch.save(encoder, model_ckpt)


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
        ngpu=args['n_gpu'],
        ally_classes=int(args['n_ally']),
        advr_1_classes=int(args['n_advr_1']),
        advr_2_classes=int(args['n_advr_2']),
        encoding_dim=args['dim'],
        hidden_dim=args['hidden_dim'],
        leaky=args['leaky'] == 1,
        activation=args['activation'],
        test_size=args['test_size'],
        batch_size=args['batch_size'],
        n_epochs=args['n_epochs'],
        shuffle=args['shuffle'] == 1,
        init_weight=args['init_w'] == 1,
        lr_encd=args['lr_encd'],
        lr_ally=args['lr_ally'],
        lr_advr_1=args['lr_advr_1'],
        lr_advr_2=args['lr_advr_2'],
        alpha=args['alpha'],
        g_reps=args['g_reps'],
        d_reps=args['d_reps'],
        expt=args['expt'],
        marker=marker
    )
    log_time('End', time_stp()[0])
    sep()
