import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as utils

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.argparser import autoencoder_argparse
from common.utility import log_shapes, log_time, torch_device,\
    time_stp, load_processed_data, logger, sep
from common.torchsummary import summary

from models.autoencoder import AutoEncoderBasic


def main(
        model,
        time_stamp,
        device,
        ally_classes,
        advr_1_classes,
        advr_2_classes,
        encoding_dim,
        test_size,
        batch_size,
        n_epochs,
        shuffle,
        lr,
        expt,
        ):

    device = torch_device(device=device)

    # refer to PrivacyGAN_Titanic for data preparation
    X, y_ally, y_advr_1, y_advr_2 = load_processed_data(
        expt, 'processed_data_X_y_ally_y_advr_y_advr_2.pkl')
    log_shapes(
        [X, y_ally, y_advr_1, y_advr_2],
        locals(),
        'Dataset loaded'
    )

    X_train, X_valid = train_test_split(
            X,
            test_size=test_size,
            stratify=pd.DataFrame(np.concatenate(
                (
                    y_ally.reshape(-1, ally_classes),
                    y_advr_1.reshape(-1, advr_1_classes),
                    y_advr_2.reshape(-1, advr_2_classes),
                ), axis=1)
            )
        )

    log_shapes(
        [
            X_train, X_valid,
        ],
        locals(),
        'Data size after train test split'
    )

    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_valid_normalized = scaler.transform(X_valid)

    log_shapes([X_train_normalized, X_valid_normalized], locals())

    dataset_train = utils.TensorDataset(torch.Tensor(X_train_normalized))
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    dataset_valid = utils.TensorDataset(torch.Tensor(X_valid_normalized))
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=2)

    auto_encoder = AutoEncoderBasic(
        input_size=X_train_normalized.shape[1],
        encoding_dim=encoding_dim
    ).to(device)

    criterion = torch.nn.MSELoss()
    adam_optim = torch.optim.Adam
    optimizer = adam_optim(auto_encoder.parameters(), lr=lr)

    summary(auto_encoder, input_size=(1, X_train_normalized.shape[1]))

    h_epoch = []
    h_valid = []
    h_train = []

    auto_encoder.train()

    sep()
    logging.info("epoch \t Aencoder_train \t Aencoder_valid")

    for epoch in range(n_epochs):

        nsamples = 0
        iloss = 0
        for data in dataloader_train:
            optimizer.zero_grad()

            X_torch = data[0].to(device)
            X_torch_hat = auto_encoder(X_torch)
            loss = criterion(X_torch_hat, X_torch)
            loss.backward()
            optimizer.step()

            nsamples += 1
            iloss += loss.item()

        if epoch % int(n_epochs/10) != 0:
            continue

        h_epoch.append(epoch)
        h_train.append(iloss/nsamples)

        nsamples = 0
        iloss = 0
        for data in dataloader_valid:
            X_torch = data[0].to(device)
            X_torch_hat = auto_encoder(X_torch)
            loss = criterion(X_torch_hat, X_torch)

            nsamples += 1
            iloss += loss.item()
        h_valid.append(iloss/nsamples)

        logging.info('{} \t {:.8f} \t {:.8f}'.format(
            h_epoch[-1],
            h_train[-1],
            h_valid[-1],
        ))

    config_summary = 'device_{}_dim_{}_batch_{}_epochs_{}_lr_{}_tr_{:.4f}_val_{:.4f}'\
        .format(
            device,
            encoding_dim,
            batch_size,
            n_epochs,
            lr,
            h_train[-1],
            h_valid[-1],
        )

    plt.plot(h_epoch, h_train, 'r--')
    plt.plot(h_epoch, h_valid, 'b--')
    plt.legend(['train_loss', 'valid_loss'])
    plt.title("autoencoder training {}".format(config_summary))

    plot_location = 'plots/{}/{}_training_{}_{}.png'.format(
        expt, model, time_stamp, config_summary)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location)
    checkpoint_location = \
        'checkpoints/{}/{}_training_history_{}_{}.pkl'.format(
            expt, model, time_stamp, config_summary)
    logging.info('Saving: {}'.format(checkpoint_location))
    pkl.dump((h_epoch, h_train, h_valid), open(checkpoint_location, 'wb'))

    model_ckpt = 'checkpoints/{}/{}_torch_model_{}_{}.pkl'.format(
            expt, model, time_stamp, config_summary)
    logging.info('Saving: {}'.format(model_ckpt))
    torch.save(auto_encoder, model_ckpt)


if __name__ == "__main__":
    expt = 'mimic'
    model = 'autoencoder_basic'
    marker = 'A'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, 'A')

    log_time('Start', pr_time)
    args = autoencoder_argparse()
    main(
        model=model,
        time_stamp=fl_time,
        device=args['device'],
        ally_classes=int(args['n_ally']),
        advr_1_classes=int(args['n_advr_1']),
        advr_2_classes=int(args['n_advr_2']),
        encoding_dim=int(args['dim']),
        test_size=float(args['test_size']),
        batch_size=int(args['batch_size']),
        n_epochs=int(args['n_epochs']),
        shuffle=int(args['shuffle']) == 1,
        lr=float(args['lr']),
        expt=args['expt'],
    )
    log_time('End', time_stp()[0])
    sep()
