import joblib
import logging
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import torch.nn as nn
import torch.utils.data as utils

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.argparser import comparison_argparse
from common.utility import log_time, torch_device,\
    time_stp, logger, sep, weights_init
from common.torchsummary import summary

from preprocessing import get_data

from models.eigan import DiscriminatorFCN

def main(
        model,
        time_stamp,
        device,
        ally_classes,
        advr_classes,
        encoding_dim,
        hidden_dim,
        leaky,
        test_size,
        batch_size,
        n_epochs,
        shuffle,
        lr_ally,
        lr_advr,
        expt,
        pca_ckpt,
        autoencoder_ckpt,
        encoder_ckpt,
        ):

    device = torch_device(device=device)

    X_normalized_train, X_normalized_valid,\
        y_ally_train, y_ally_valid, \
        y_advr_train, y_advr_valid, = get_data(expt, test_size)

    pca = joblib.load(pca_ckpt)
    autoencoder = torch.load(autoencoder_ckpt)
    autoencoder.eval()
    encoder = torch.load(encoder_ckpt)
    encoder.eval()

    optim = torch.optim.Adam
    criterionBCEWithLogits = nn.BCEWithLogitsLoss()
    criterionCrossEntropy = nn.CrossEntropyLoss()

    h = {
        'epoch': {
            'train': [],
            'valid': [],
        },
        'pca': {
            'ally_train': [],
            'ally_valid': [],
            'advr_train': [],
            'advr_valid': [],
        },
        'autoencoder': {
            'ally_train': [],
            'ally_valid': [],
            'advr_train': [],
            'advr_valid': [],
        },
        'encoder': {
            'ally_train': [],
            'ally_valid': [],
            'advr_train': [],
            'advr_valid': [],
        },
    }

    for _ in ['encoder']:
        if _ == 'pca':
            dataset_train = utils.TensorDataset(
                torch.Tensor(pca.eval(X_normalized_train)),
                torch.Tensor(y_ally_train),
                torch.Tensor(y_advr_train)
            )

            dataset_valid = utils.TensorDataset(
                torch.Tensor(pca.eval(X_normalized_valid)),
                torch.Tensor(y_ally_valid),
                torch.Tensor(y_advr_valid)
            )

            def transform(input_arg):
                return input_arg
        else:
            dataset_train = utils.TensorDataset(
                torch.Tensor(X_normalized_train),
                torch.Tensor(y_ally_train),
                torch.Tensor(y_advr_train)
            )

            dataset_valid = utils.TensorDataset(
                torch.Tensor(X_normalized_valid),
                torch.Tensor(y_ally_valid),
                torch.Tensor(y_advr_valid)
            )
            if _ == 'autoencoder':
                def transform(input_arg):
                    return autoencoder.encoder(input_arg)

            elif _ == 'encoder':
                def transform(input_arg):
                    return encoder(input_arg)

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
        advr = DiscriminatorFCN(
            encoding_dim, hidden_dim, advr_classes,
            leaky).to(device)

        ally.apply(weights_init)
        advr.apply(weights_init)

        sep('{}:{}'.format(_, 'ally'))
        summary(ally, input_size=(1, encoding_dim))
        sep('{}:{}'.format(_, 'advr'))
        summary(advr, input_size=(1, encoding_dim))

        optimizer_ally = optim(ally.parameters(), lr=lr_ally)
        optimizer_advr = optim(advr.parameters(), lr=lr_advr)

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

            h['epoch']['train'].append(epoch)
            h[_]['ally_train'].append(iloss_ally/nsamples)

            if epoch % int(n_epochs/10) != 0:
                continue

            ally.eval()

            nsamples = 0
            iloss_ally = 0

            for i, data in enumerate(dataloader_valid, 0):
                X_valid_torch = transform(data[0].to(device))
                y_ally_valid_torch = data[1].to(device)
                y_ally_valid_hat_torch = ally(X_valid_torch)

                valid_loss_ally = criterionBCEWithLogits(
                    y_ally_valid_hat_torch, y_ally_valid_torch)

                nsamples += 1
                iloss_ally += valid_loss_ally.item()

            h['epoch']['valid'].append(epoch)
            h[_]['ally_valid'].append(iloss_ally/nsamples)

            logging.info(
                '{} \t {:.8f} \t {:.8f}'.
                format(
                    epoch,
                    h[_]['ally_train'][-1],
                    h[_]['ally_valid'][-1],
                ))

        # adversary
        sep("adversary")
        logging.info('{} \t {} \t {}'.format(
            'Epoch',
            'Advr Train',
            'Advr Valid',
            ))

        for epoch in range(n_epochs):
            advr.train()

            nsamples = 0
            iloss_advr = 0
            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = transform(data[0].to(device))
                y_advr_train_torch = data[2].to(device)

                optimizer_advr.zero_grad()
                y_advr_train_hat_torch = advr(X_train_torch)

                loss_advr = criterionCrossEntropy(
                    y_advr_train_hat_torch,
                    torch.argmax(y_advr_train_torch, 1))
                loss_advr.backward()
                optimizer_advr.step()

                nsamples += 1
                iloss_advr += loss_advr.item()

            h[_]['advr_train'].append(iloss_advr/nsamples)

            if epoch % int(n_epochs/10) != 0:
                continue

            advr.eval()

            nsamples = 0
            iloss_advr = 0

            for i, data in enumerate(dataloader_valid, 0):
                X_valid_torch = transform(data[0].to(device))
                y_advr_valid_torch = data[2].to(device)
                y_advr_valid_hat_torch = advr(X_valid_torch)

                valid_loss_advr = criterionCrossEntropy(
                    y_advr_valid_hat_torch, torch.argmax(y_advr_valid_torch, 1))

                nsamples += 1
                iloss_advr += valid_loss_advr.item()

            h[_]['advr_valid'].append(iloss_advr/nsamples)

            logging.info(
                '{} \t {:.8f} \t {:.8f}'.
                format(
                    epoch,
                    h[_]['advr_train'][-1],
                    h[_]['advr_valid'][-1],
                ))

    checkpoint_location = \
        'checkpoints/{}/{}_training_history_{}.pkl'.format(
            expt, model, time_stamp)
    sep()
    logging.info('Saving: {}'.format(checkpoint_location))
    pkl.dump(h, open(checkpoint_location, 'wb'))

    plt.plot(h['epoch']['valid'], h['encoder']['ally_valid'], 'r')
    plt.plot(h['epoch']['valid'], h['encoder']['advr_valid'], 'r--')
    plt.plot(h['epoch']['valid'], h['autoencoder']['ally_valid'], 'b')
    plt.plot(h['epoch']['valid'], h['autoencoder']['advr_valid'], 'b--')
    plt.plot(h['epoch']['valid'], h['pca']['ally_valid'], 'g')
    plt.plot(h['epoch']['valid'], h['pca']['advr_valid'], 'g--')
    plt.legend([
        'gan ally', 'gan advr',
        'autoencoder ally', 'autoencoder advr',
        'pca ally', 'pca advr',
    ])

    plt.title("{} on {} training".format(model, expt))

    plot_location = 'plots/{}/{}_training_{}.png'.format(
        expt, model, time_stamp)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location)


if __name__ == "__main__":
    expt = 'mnist'
    model = 'comparison'
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
        advr_classes=args['n_advr'],
        encoding_dim=args['dim'],
        hidden_dim=args['hidden_dim'],
        leaky=args['leaky'],
        test_size=args['test_size'],
        batch_size=args['batch_size'],
        n_epochs=args['n_epochs'],
        shuffle=args['shuffle'] == 1,
        lr_ally=args['lr_ally'],
        lr_advr=args['lr_advr'],
        expt=args['expt'],
        pca_ckpt=args['pca_ckpt'],
        autoencoder_ckpt=args['autoencoder_ckpt'],
        encoder_ckpt=args['encoder_ckpt']
    )
    log_time('End', time_stp()[0])
    sep()
