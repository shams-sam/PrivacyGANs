import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

from models.eigan import GeneratorCNN, DiscriminatorFCN


def main(
        model,
        time_stamp,
        device,
        ngpu,
        ally_classes,
        advr_classes,
        num_channels,
        num_filters,
        encoding_dim,
        hidden_dim,
        test_size,
        batch_size,
        n_epochs,
        shuffle,
        init_weight,
        lr_encd,
        lr_ally,
        lr_advr,
        alpha,
        g_reps,
        d_reps,
        expt,
        ):

    device = torch_device(device=device)

    X, y = load_processed_data(expt, 'processed_data_X_y.pkl')
    log_shapes(
        [X, y],
        locals(),
        'Dataset loaded'
    )

    y_ally = y % 2
    y_advr = y

    X_train, X_valid, \
        y_ally_train, y_ally_valid, \
        y_advr_train, y_advr_valid = train_test_split(
            X,
            y_ally,
            y_advr,
            test_size=test_size,
            stratify=pd.DataFrame(np.concatenate(
                (
                    y_ally.reshape(-1, 1),
                    y_advr.reshape(-1, 1),
                ), axis=1)
            )
        )

    scaler = MinMaxScaler()
    scaler.fit(X_train.astype(np.float64))
    X_normalized_train = scaler.transform(X_train.astype(np.float64))
    X_normalized_valid = scaler.transform(X_valid.astype(np.float64))

    y_ally_train = y_ally_train.reshape(-1, 1)
    y_ally_valid = y_ally_valid.reshape(-1, 1)
    y_advr_train = to_categorical(y_advr_train)
    y_advr_valid = to_categorical(y_advr_valid)

    log_shapes(
        [
            X_normalized_train, X_normalized_valid,
            y_ally_train, y_ally_valid,
            y_advr_train, y_advr_valid
        ],
        locals(),
        'Data size after train test split'
    )

    encoder = GeneratorCNN(
        ngpu, num_channels,
        num_filters, encoding_dim).to(device)
    ally = DiscriminatorFCN(encoding_dim, hidden_dim, ally_classes).to(device)
    advr = DiscriminatorFCN(encoding_dim, hidden_dim, advr_classes).to(device)

    if init_weight:
        sep()
        logging.info('applying weights_init ...')
        encoder.apply(weights_init)
        ally.apply(weights_init)
        advr.apply(weights_init)

    sep('encoder')
    summary(encoder, input_size=(1, 28, 28))
    sep('ally')
    summary(ally, input_size=(1, encoding_dim))
    sep('advr')
    summary(advr, input_size=(1, encoding_dim))

    optim = torch.optim.Adam
    criterionBCEWithLogits = nn.BCEWithLogitsLoss()
    criterionCrossEntropy = nn.CrossEntropyLoss()

    optimizer_encd = optim(encoder.parameters(), lr=lr_encd)
    optimizer_ally = optim(ally.parameters(), lr=lr_ally)
    optimizer_advr = optim(advr.parameters(), lr=lr_advr)

    dataset_train = utils.TensorDataset(
        torch.Tensor(X_normalized_train.reshape(-1, 1, 28, 28)),
        torch.Tensor(y_ally_train),
        torch.Tensor(y_advr_train)
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )

    dataset_valid = utils.TensorDataset(
        torch.Tensor(X_normalized_valid.reshape(-1, 1, 28, 28)),
        torch.Tensor(y_ally_valid),
        torch.Tensor(y_advr_valid)
    )

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )

    epochs_train = []
    epochs_valid = []
    encd_loss_train = []
    encd_loss_valid = []
    ally_loss_train = []
    ally_loss_valid = []
    advr_loss_train = []
    advr_loss_valid = []

    logging.info('{} \t {} \t {} \t {} \t {} \t {} \t {}'.format(
                'Epoch',
                'Encd Train',
                'Encd Valid',
                'Ally Train',
                'Ally Valid',
                'Advr Train',
                'Advr Valid',
                ))

    for epoch in range(n_epochs):

        encoder.train()
        ally.eval()
        advr.eval()

        for __ in range(g_reps):
            nsamples = 0
            iloss = 0
            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = data[0].to(device)
                y_ally_train_torch = data[1].to(device)
                y_advr_train_torch = data[2].to(device)

                optimizer_encd.zero_grad()
                # Forward pass
                X_train_encoded = encoder(X_train_torch)
                y_ally_train_hat_torch = ally(X_train_encoded)
                y_advr_train_hat_torch = advr(X_train_encoded)
                # Compute Loss
                loss_ally = criterionBCEWithLogits(
                    y_ally_train_hat_torch, y_ally_train_torch)
                loss_advr = criterionCrossEntropy(
                    y_advr_train_hat_torch,
                    torch.argmax(y_advr_train_torch, 1))
                loss_encd = alpha * loss_ally - (1-alpha) * loss_advr
                # Backward pass
                loss_encd.backward()
                optimizer_encd.step()

                nsamples += 1
                iloss += loss_encd.item()

        epochs_train.append(epoch)
        encd_loss_train.append(iloss/nsamples)

        encoder.eval()
        ally.train()
        advr.train()

        for __ in range(d_reps):
            nsamples = 0
            iloss_ally = 0
            iloss_advr = 0
            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = data[0].to(device)
                y_ally_train_torch = data[1].to(device)
                y_advr_train_torch = data[2].to(device)

                optimizer_ally.zero_grad()
                X_train_encoded = encoder(X_train_torch)
                y_ally_train_hat_torch = ally(X_train_encoded)
                loss_ally = criterionBCEWithLogits(
                    y_ally_train_hat_torch, y_ally_train_torch)
                loss_ally.backward()
                optimizer_ally.step()

                optimizer_advr.zero_grad()
                X_train_encoded = encoder(X_train_torch)
                y_advr_train_hat_torch = advr(X_train_encoded)
                loss_advr = criterionCrossEntropy(
                    y_advr_train_hat_torch,
                    torch.argmax(y_advr_train_torch, 1))
                loss_advr.backward()
                optimizer_advr.step()

                nsamples += 1
                iloss_ally += loss_ally.item()
                iloss_advr += loss_advr.item()

        ally_loss_train.append(iloss_ally/nsamples)
        advr_loss_train.append(iloss_advr/nsamples)

        if epoch % int(n_epochs/10) != 0:
            continue

        encoder.eval()
        ally.eval()
        advr.eval()

        nsamples = 0
        iloss = 0
        iloss_ally = 0
        iloss_advr = 0

        for i, data in enumerate(dataloader_valid, 0):
            X_valid_torch = data[0].to(device)
            y_ally_valid_torch = data[1].to(device)
            y_advr_valid_torch = data[2].to(device)
            X_valid_encoded = encoder(X_valid_torch)
            y_ally_valid_hat_torch = ally(X_valid_encoded)
            y_advr_valid_hat_torch = advr(X_valid_encoded)

            valid_loss_ally = criterionBCEWithLogits(
                y_ally_valid_hat_torch, y_ally_valid_torch)
            valid_loss_advr = criterionCrossEntropy(
                y_advr_valid_hat_torch, torch.argmax(y_advr_valid_torch, 1))
            valid_loss_encd = alpha * valid_loss_ally - \
                (1-alpha) * valid_loss_advr

            nsamples += 1
            iloss += valid_loss_encd.item()
            iloss_ally += valid_loss_ally.item()
            iloss_advr += valid_loss_advr.item()

        epochs_valid.append(epoch)
        encd_loss_valid.append(iloss/nsamples)
        ally_loss_valid.append(iloss_ally/nsamples)
        advr_loss_valid.append(iloss_advr/nsamples)

        logging.info(
            '{} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f}'.
            format(
                epoch,
                encd_loss_train[-1],
                encd_loss_valid[-1],
                ally_loss_train[-1],
                ally_loss_valid[-1],
                advr_loss_train[-1],
                advr_loss_valid[-1],
            ))

    config_summary = 'device_{}_dim_{}_hidden_{}_batch_{}_nc_{}_ndf_{}_\
        epochs_{}_lrencd_{}_lrally_{}_lradvr_{}_tr_{:.4f}_val_{:.4f}'\
        .format(
            device,
            encoding_dim,
            hidden_dim,
            batch_size,
            num_channels,
            num_filters,
            n_epochs,
            lr_encd,
            lr_ally,
            lr_advr,
            encd_loss_train[-1],
            advr_loss_valid[-1],
        )

    plt.plot(epochs_train, encd_loss_train, 'r')
    plt.plot(epochs_valid, encd_loss_valid, 'r--')
    plt.plot(epochs_train, ally_loss_train, 'b')
    plt.plot(epochs_valid, ally_loss_valid, 'b--')
    plt.plot(epochs_train, advr_loss_train, 'g')
    plt.plot(epochs_valid, advr_loss_valid, 'g--')
    plt.legend([
        'encoder train', 'encoder valid',
        'ally train', 'ally valid',
        'advr train', 'advr valid',
    ])
    plt.title("{} on {} training {}".format(model, expt, config_summary))

    plot_location = 'plots/{}/{}_training_{}_{}.png'.format(
        expt, model, time_stamp, config_summary)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location)
    # checkpoint_location = \
    #     'checkpoints/{}/{}_training_history_{}_{}.pkl'.format(
    #         expt, model, time_stamp, config_summary)
    # logging.info('Saving: {}'.format(checkpoint_location))
    # pkl.dump((h_epoch, h_train, h_valid), open(checkpoint_location, 'wb'))

    # model_ckpt = 'checkpoints/{}/{}_torch_model_{}_{}.pkl'.format(
    #         expt, model, time_stamp, config_summary)
    # logging.info('Saving: {}'.format(model_ckpt))
    # torch.save(auto_encoder, model_ckpt)


if __name__ == "__main__":
    expt = 'mnist'
    model = 'eigan'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time)

    log_time('Start', pr_time)
    args = eigan_argparse()
    main(
        model=model,
        time_stamp=fl_time,
        device=args['device'],
        ngpu=args['n_gpu'],
        ally_classes=int(args['n_ally']),
        advr_classes=int(args['n_advr']),
        num_channels=args['n_channels'],
        num_filters=args['n_filters'],
        encoding_dim=args['dim'],
        hidden_dim=args['hidden_dim'],
        test_size=args['test_size'],
        batch_size=args['batch_size'],
        n_epochs=args['n_epochs'],
        shuffle=args['shuffle'] == 1,
        init_weight=args['init_w'] == 1,
        lr_encd=args['lr_encd'],
        lr_ally=args['lr_ally'],
        lr_advr=args['lr_advr'],
        alpha=args['alpha'],
        g_reps=args['g_reps'],
        d_reps=args['d_reps'],
        expt=args['expt'],
    )
    log_time('End', time_stp()[0])
    sep()
