import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as utils
from torchsummary import summary

matplotlib.use('Agg')

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.autoencoder_basic import AutoEncoder
from common.utility import print_shapes, print_time, torch_device,\
    time_stp, load_processed_data


def main(
        device,
        ally_classes,
        advr_classes,
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
    X, y_ally, y_advr = load_processed_data(expt)
    print_shapes(
        [X, y_ally, y_advr],
        locals(),
        'Dataset loaded'
    )

    X_train, X_valid, \
        y_ally_train, y_ally_valid, \
        y_advr_train, y_advr_valid = train_test_split(
            X,
            y_ally,
            y_advr,
            test_size=test_size,
            stratify=pd.DataFrame(np.concatenate(
                (
                    y_ally.reshape(-1, ally_classes),
                    y_advr.reshape(-1, advr_classes),
                ), axis=1)
            )
        )

    print_shapes(
        [
            X_train, X_valid,
            y_ally_train, y_ally_valid,
            y_advr_train, y_advr_valid
        ],
        locals(),
        'Data size after train test split'
    )

    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_valid_normalized = scaler.transform(X_valid)

    print_shapes([X_train_normalized, X_valid_normalized], locals())

    dataset_train = utils.TensorDataset(torch.Tensor(X_train_normalized))
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    dataset_valid = utils.TensorDataset(torch.Tensor(X_valid_normalized))
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=2)

    auto_encoder = AutoEncoder(
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

    print("epoch \t Aencoder_train \t Aencoder_valid")

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

        print('{} \t {:.8f} \t {:.8f}'.format(
            h_epoch[-1],
            h_train[-1],
            h_valid[-1],
        ))

    config_summary = '{}_dim_{}_batch_{}_epochs_{}_lr_{}_tr_{:.4f}_val_{:.4f}'\
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

    plot_location = 'plots/{}/auto_encoder_training_{}_{}.png'.format(
        expt, time_stp(), config_summary)
    print('Saving: {}'.format(plot_location))
    plt.savefig(plot_location)
    checkpoint_location = \
        'checkpoints/{}/auto_encoder_training_history_{}_{}.pkl'.format(
            expt, time_stp(), config_summary)
    print('Saving: {}'.format(checkpoint_location))
    pkl.dump((h_epoch, h_train, h_valid), open(checkpoint_location, 'wb'))


if __name__ == "__main__":
    print_time('Start')
    main(
        device='cpu',
        ally_classes=1,
        advr_classes=1,
        encoding_dim = 588,
        test_size=0.1,
        batch_size=1024,
        n_epochs=101,
        shuffle=True,
        lr=0.0001,
        expt='mimic',
    )
    print_time('End')
    print('='*80)
