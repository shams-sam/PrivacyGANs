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
    time_stp, logger, sep, weights_init, load_processed_data
from common.torchsummary import summary

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
        test_size,
        batch_size,
        n_epochs,
        shuffle,
        lr,
        expt,
        pca_ckpt,
        autoencoder_ckpt,
        encoder_ckpt,
        ):
    device = torch_device(device=device)

    X, targets = load_processed_data(
        expt, 'processed_data_X_targets.pkl')
    log_shapes(
        [X] + [targets[i] for i in targets],
        locals(),
        'Dataset loaded'
    )

    h = {}

    for name, target in targets.items():

        sep(name)

        target = target.reshape(-1, 1)

        X_train, X_valid, \
            y_train, y_valid = train_test_split(
                X,
                target,
                test_size=test_size,
                stratify=target
            )

        log_shapes(
            [
                X_train, X_valid,
                y_train, y_valid,
            ],
            locals(),
            'Data size after train test split'
        )

        scaler = StandardScaler()
        X_normalized_train = scaler.fit_transform(X_train)
        X_normalized_valid = scaler.transform(X_valid)

        log_shapes([X_normalized_train, X_normalized_valid], locals())

        optim = torch.optim.Adam
        criterionBCEWithLogits = nn.BCEWithLogitsLoss()

        h[name] = {
                'epoch_train': [],
                'epoch_valid': [],
                'y_train': [],
                'y_valid': [],
        }

        dataset_train = utils.TensorDataset(
            torch.Tensor(X_normalized_train),
            torch.Tensor(y_train.reshape(-1, 1))
        )

        dataset_valid = utils.TensorDataset(
            torch.Tensor(X_normalized_valid),
            torch.Tensor(y_valid.reshape(-1, 1)),
        )

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

        clf = DiscriminatorFCN(
            encoding_dim, hidden_dim, 1,
            leaky).to(device)

        clf.apply(weights_init)

        sep('{}:{}'.format(name, 'summary'))
        summary(clf, input_size=(1, encoding_dim))

        optimizer = optim(clf.parameters(), lr=lr)

        # adversary 1
        sep("TRAINING")
        logging.info('{} \t {} \t {}'.format(
            'Epoch',
            'Train',
            'Valid',
            ))

        for epoch in range(n_epochs):

            clf.train()

            nsamples = 0
            iloss_train = 0

            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = data[0].to(device)
                y_train_torch = data[1].to(device)

                optimizer.zero_grad()
                y_train_hat_torch = clf(X_train_torch)

                loss_train = criterionBCEWithLogits(
                    y_train_hat_torch, y_train_torch)
                loss_train.backward()
                optimizer.step()

                nsamples += 1
                iloss_train += loss_train.item()

            h[name]['y_train'].append(iloss_train/nsamples)
            h[name]['epoch_train'].append(epoch)

            if epoch % int(n_epochs/10) != 0:
                continue

            clf.eval()

            nsamples = 0
            iloss_valid = 0
            correct = 0
            total = 0

            for i, data in enumerate(dataloader_valid, 0):
                X_valid_torch = data[0].to(device)
                y_valid_torch = data[1].to(device)
                y_valid_hat_torch = clf(X_valid_torch)

                valid_loss = criterionBCEWithLogits(
                    y_valid_hat_torch, y_valid_torch,)

                predicted = y_valid_hat_torch > 0.5

                nsamples += 1
                iloss_valid += valid_loss.item()
                total += y_valid_torch.size(0)
                correct += (predicted == y_valid_torch).sum().item()

            h[name]['y_valid'].append(iloss_valid/nsamples)
            h[name]['epoch_valid'].append(epoch)

            logging.info(
                '{} \t {:.8f} \t {:.8f} \t {:.8f}'.
                format(
                    epoch,
                    h[name]['y_train'][-1],
                    h[name]['y_valid'][-1],
                    correct/total
                ))

    checkpoint_location = \
        'checkpoints/{}/{}_training_history_{}.pkl'.format(
            expt, model, time_stamp)
    sep()
    logging.info('Saving: {}'.format(checkpoint_location))
    pkl.dump(h, open(checkpoint_location, 'wb'))


if __name__ == "__main__":
    expt = 'mimic'
    model = 'n_ind'
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
        test_size=args['test_size'],
        batch_size=args['batch_size'],
        n_epochs=args['n_epochs'],
        shuffle=args['shuffle'] == 1,
        lr=args['lr'],
        expt=args['expt'],
        pca_ckpt=args['pca_ckpt'],
        autoencoder_ckpt=args['autoencoder_ckpt'],
        encoder_ckpt=args['encoder_ckpt']
    )
    log_time('End', time_stp()[0])
    sep()
