import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.utils.data as utils

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.argparser import comparison_argparse
from common.utility import log_shapes, log_time, torch_device,\
    time_stp, logger, sep, weights_init, load_processed_data, laplaceMechanism
from common.torchsummary import summary

from preprocessing import get_data

from models.eigan import DiscriminatorFCN


def main(
        model,
        time_stamp,
        device,
        ally_classes,
        advr_1_classes,
        advr_2_classes,
        encoding_dim,
        hidden_dim,
        leaky,
        epsilon,
        test_size,
        batch_size,
        n_epochs,
        shuffle,
        lr_ally,
        lr_advr_1,
        lr_advr_2,
        expt,
        pca_ckpt,
        autoencoder_ckpt,
        encoder_ckpt,
        ):
    device = torch_device(device=device)

    X_normalized_train, X_normalized_valid,\
        y_ally_train, y_ally_valid, \
        y_advr_1_train, y_advr_1_valid, \
        y_advr_2_train, y_advr_2_valid = get_data(expt, test_size)

    optim = torch.optim.Adam
    criterionBCEWithLogits = nn.BCEWithLogitsLoss()
    criterionCrossEntropy = nn.CrossEntropyLoss()

    h = {
        'epoch': {
            'train': [],
            'valid': [],
        },
        'dp': {
            'ally_train': [],
            'ally_valid': [],
            'advr_1_train': [],
            'advr_1_valid': [],
            'advr_2_train': [],
            'advr_2_valid': [],
        },
    }

    for _ in ['dp']:

        dataset_train = utils.TensorDataset(
            torch.Tensor(X_normalized_train),
            torch.Tensor(y_ally_train.reshape(-1, ally_classes)),
            torch.Tensor(y_advr_1_train.reshape(-1, advr_1_classes)),
            torch.Tensor(y_advr_2_train.reshape(-1, advr_2_classes)),
        )

        dataset_valid = utils.TensorDataset(
            torch.Tensor(X_normalized_valid),
            torch.Tensor(y_ally_valid.reshape(-1, ally_classes)),
            torch.Tensor(y_advr_1_valid.reshape(-1, advr_1_classes)),
            torch.Tensor(y_advr_2_valid.reshape(-1, advr_2_classes)),
        )

        def transform(input_arg):
            return laplaceMechanism(input_arg, epsilon, device)

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=1
        )

        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=1
        )

        ally = DiscriminatorFCN(
            encoding_dim, hidden_dim, ally_classes,
            leaky).to(device)
        advr_1 = DiscriminatorFCN(
            encoding_dim, hidden_dim, advr_1_classes,
            leaky).to(device)
        advr_2 = DiscriminatorFCN(
            encoding_dim, hidden_dim, advr_2_classes,
            leaky).to(device)

        ally.apply(weights_init)
        advr_1.apply(weights_init)
        advr_2.apply(weights_init)

        sep('{}:{}'.format(_, 'ally'))
        summary(ally, input_size=(1, encoding_dim))
        sep('{}:{}'.format(_, 'advr 1'))
        summary(advr_1, input_size=(1, encoding_dim))
        sep('{}:{}'.format(_, 'advr 2'))
        summary(advr_2, input_size=(1, encoding_dim))

        optimizer_ally = optim(ally.parameters(), lr=lr_ally)
        optimizer_advr_1 = optim(advr_1.parameters(), lr=lr_advr_1)
        optimizer_advr_2 = optim(advr_2.parameters(), lr=lr_advr_2)

        

        # adversary 1
        sep("adversary 1")
        logging.info('{} \t {} \t {}'.format(
            'Epoch',
            'Advr 1 Train',
            'Advr 1 Valid',
            ))

        for epoch in range(n_epochs):
            advr_1.train()

            nsamples = 0
            iloss_advr = 0
            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = transform(data[0].to(device))
                y_advr_train_torch = data[2].to(device)

                optimizer_advr_1.zero_grad()
                y_advr_train_hat_torch = advr_1(X_train_torch)

                loss_advr = criterionCrossEntropy(
                    y_advr_train_hat_torch,
                    torch.argmax(y_advr_train_torch, 1))
                loss_advr.backward()
                optimizer_advr_1.step()

                nsamples += 1
                iloss_advr += loss_advr.item()

            h[_]['advr_1_train'].append(iloss_advr/nsamples)

            if epoch % int(n_epochs/10) != 0:
                continue

            advr_1.eval()

            nsamples = 0
            iloss_advr = 0
            correct = 0
            total = 0

            for i, data in enumerate(dataloader_valid, 0):
                X_valid_torch = transform(data[0].to(device))
                y_advr_valid_torch = data[2].to(device)
                y_advr_valid_hat_torch = advr_1(X_valid_torch)

                valid_loss_advr = criterionCrossEntropy(
                    y_advr_valid_hat_torch,
                    torch.argmax(y_advr_valid_torch, 1))

                tmp, predicted = torch.max(y_advr_valid_hat_torch, 1)
                tmp, actual = torch.max(y_advr_valid_torch, 1) 

                nsamples += 1
                iloss_advr += valid_loss_advr.item()
                total += actual.size(0)
                correct += (predicted == actual).sum().item()

            h[_]['advr_1_valid'].append(iloss_advr/nsamples)

            logging.info(
                '{} \t {:.8f} \t {:.8f} \t {:.8f}'.
                format(
                    epoch,
                    h[_]['advr_1_train'][-1],
                    h[_]['advr_1_valid'][-1],
                    correct/total
                ))

        # adversary
        sep("adversary 2")
        logging.info('{} \t {} \t {}'.format(
            'Epoch',
            'Advr 2 Train',
            'Advr 2 Valid',
            ))

        for epoch in range(n_epochs):
            advr_2.train()

            nsamples = 0
            iloss_advr = 0
            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = transform(data[0].to(device))
                y_advr_train_torch = data[3].to(device)

                optimizer_advr_2.zero_grad()
                y_advr_train_hat_torch = advr_2(X_train_torch)

                loss_advr = criterionBCEWithLogits(
                    y_advr_train_hat_torch, y_advr_train_torch)
                loss_advr.backward()
                optimizer_advr_2.step()

                nsamples += 1
                iloss_advr += loss_advr.item()

            h[_]['advr_2_train'].append(iloss_advr/nsamples)

            if epoch % int(n_epochs/10) != 0:
                continue

            advr_2.eval()

            nsamples = 0
            iloss_advr = 0
            correct = 0
            total = 0

            for i, data in enumerate(dataloader_valid, 0):
                X_valid_torch = transform(data[0].to(device))
                y_advr_valid_torch = data[3].to(device)
                y_advr_valid_hat_torch = advr_2(X_valid_torch)

                valid_loss_advr = criterionBCEWithLogits(
                    y_advr_valid_hat_torch, y_advr_valid_torch,)

                predicted = y_advr_valid_hat_torch > 0.5

                nsamples += 1
                iloss_advr += valid_loss_advr.item()
                total += y_advr_valid_torch.size(0)
                correct += (predicted == y_advr_valid_torch).sum().item()

            h[_]['advr_2_valid'].append(iloss_advr/nsamples)

            logging.info(
                '{} \t {:.8f} \t {:.8f} \t {:.8f}'.
                format(
                    epoch,
                    h[_]['advr_2_train'][-1],
                    h[_]['advr_2_valid'][-1],
                    correct/total
                ))

        sep("ally")
        logging.info('{} \t {} \t {}'.format(
            'Epoch',
            'Ally Train',
            'Ally Valid',
            ))

        for epoch in range(n_epochs):
            ally.train()

            nsamples = 0
            iloss_ally = 0
            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = transform(data[0].to(device))
                y_ally_train_torch = data[1].to(device)

                optimizer_ally.zero_grad()
                y_ally_train_hat_torch = ally(X_train_torch)
                loss_ally = criterionBCEWithLogits(
                    y_ally_train_hat_torch, y_ally_train_torch)
                loss_ally.backward()
                optimizer_ally.step()

                nsamples += 1
                iloss_ally += loss_ally.item()
            if epoch not in h['epoch']['train']:
                h['epoch']['train'].append(epoch)
            h[_]['ally_train'].append(iloss_ally/nsamples)

            if epoch % int(n_epochs/10) != 0:
                continue

            ally.eval()

            nsamples = 0
            iloss_ally = 0
            correct = 0
            total = 0

            for i, data in enumerate(dataloader_valid, 0):
                X_valid_torch = transform(data[0].to(device))
                y_ally_valid_torch = data[1].to(device)
                y_ally_valid_hat_torch = ally(X_valid_torch)

                valid_loss_ally = criterionBCEWithLogits(
                    y_ally_valid_hat_torch, y_ally_valid_torch)

                predicted = y_ally_valid_hat_torch > 0.5


                nsamples += 1
                iloss_ally += valid_loss_ally.item()
                total += y_ally_valid_torch.size(0)
                correct += (predicted == y_ally_valid_torch).sum().item()

            if epoch not in h['epoch']['valid']:
                h['epoch']['valid'].append(epoch)
            h[_]['ally_valid'].append(iloss_ally/nsamples)

            logging.info(
                '{} \t {:.8f} \t {:.8f} \t {:.8f}'.
                format(
                    epoch,
                    h[_]['ally_train'][-1],
                    h[_]['ally_valid'][-1],
                    correct/total
                ))

    checkpoint_location = \
        'checkpoints/{}/{}_training_history_{}.pkl'.format(
            expt, model, time_stamp)
    sep()
    logging.info('Saving: {}'.format(checkpoint_location))
    pkl.dump(h, open(checkpoint_location, 'wb'))


if __name__ == "__main__":
    expt = 'mnist'
    model = 'ind_dp'
    marker = 'A'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    args = comparison_argparse()
    main(
        model=model,
        time_stamp=fl_time,
        device=args['device'],
        ally_classes=args['n_ally'],
        advr_1_classes=args['n_advr_1'],
        advr_2_classes=args['n_advr_2'],
        encoding_dim=args['dim'],
        hidden_dim=args['hidden_dim'],
        leaky=args['leaky'],
        epsilon=args['epsilon'],
        test_size=args['test_size'],
        batch_size=args['batch_size'],
        n_epochs=args['n_epochs'],
        shuffle=args['shuffle'] == 1,
        lr_ally=args['lr_ally'],
        lr_advr_1=args['lr_advr_1'],
        lr_advr_2=args['lr_advr_2'],
        expt=args['expt'],
        pca_ckpt=args['pca_ckpt'],
        autoencoder_ckpt=args['autoencoder_ckpt'],
        encoder_ckpt=args['encoder_ckpt']
    )
    log_time('End', time_stp()[0])
    sep()
