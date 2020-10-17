import logging
import math
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import torch.utils.data as utils

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")  # noqa

from common.argparser import autoencoder_argparse
from common.utility import log_time, torch_device,\
    time_stp, logger, sep
from common.torchsummary import summary
import common.config as cfg
from preprocessing import get_data
from models.autoencoder import AutoEncoderBasic


def main(
        model,
        device,
        encoding_dim,
        batch_size,
        n_epochs,
        shuffle,
        lr,
        expt,
):

    device = torch_device(device=device)

    X_train, X_valid, \
        y_train, y_valid = get_data(expt)

    dataset_train = utils.TensorDataset(
        torch.Tensor(X_train.reshape(cfg.num_trains[expt], -1)))
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    dataset_valid = utils.TensorDataset(
        torch.Tensor(X_valid.reshape(cfg.num_tests[expt], -1)))
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=2)

    auto_encoder = AutoEncoderBasic(
        input_size=cfg.input_sizes[expt],
        encoding_dim=encoding_dim
    ).to(device)

    criterion = torch.nn.MSELoss()
    adam_optim = torch.optim.Adam
    optimizer = adam_optim(auto_encoder.parameters(), lr=lr)

    summary(auto_encoder, input_size=(1, cfg.input_sizes[expt]))

    h_epoch = []
    h_valid = []
    h_train = []

    auto_encoder.train()

    sep()
    logging.info("epoch \t train \t valid")

    best = math.inf
    config_summary = 'device_{}_dim_{}_batch_{}_epochs_{}_lr_{}'.format(
        device,
        encoding_dim,
        batch_size,
        n_epochs,
        lr,
    )

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
        if h_valid[-1] < best:
            best = h_valid[-1]

            model_ckpt = 'ckpts/{}/models/{}_{}_{}.best'.format(
                expt, model, config_summary, marker)
            logging.info('Saving: {}'.format(model_ckpt))
            torch.save(auto_encoder.state_dict(), model_ckpt)

        logging.info('{} \t {:.8f} \t {:.8f}'.format(
            h_epoch[-1],
            h_train[-1],
            h_valid[-1],
        ))

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.plot(h_epoch, h_train, 'r.:')
    ax.plot(h_epoch, h_valid, 'rs-.')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss (MSEE)')
    plt.legend(['train loss', 'valid loss'])

    plot_location = 'ckpts/{}/plots/{}_{}_{}.png'.format(
        expt, model, config_summary, marker)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location)
    checkpoint_location = 'ckpts/{}/history/{}_{}_{}.pkl'.format(
        expt, model, config_summary, marker)
    logging.info('Saving: {}'.format(checkpoint_location))
    pkl.dump((h_epoch, h_train, h_valid), open(checkpoint_location, 'wb'))

    model_ckpt = 'ckpts/{}/models/{}_{}_{}.stop'.format(
        expt, model, config_summary, marker)
    logging.info('Saving: {}'.format(model_ckpt))
    torch.save(auto_encoder.state_dict(), model_ckpt)


if __name__ == "__main__":
    expt = 'mnist'
    model = 'autoencoder_basic'
    marker = 'A'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    args = autoencoder_argparse()
    main(
        model=model,
        device=args['device'],
        encoding_dim=args['dim'],
        batch_size=args['batch_size'],
        n_epochs=args['n_epochs'],
        shuffle=args['shuffle'] == 1,
        lr=args['lr'],
        expt=args['expt'],
    )
    log_time('End', time_stp()[0])
    sep()
