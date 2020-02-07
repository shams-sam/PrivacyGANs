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

from common.argparser import eigan_argparse
from common.utility import log_shapes, log_time, torch_device,\
    time_stp, load_processed_data, logger, sep, weights_init
from common.torchsummary import summary

from models.eigan import GeneratorFCN, DiscriminatorFCN


def main(
        model,
        time_stamp,
        device,
        ngpu,
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
        lr_advr,
        alpha,
        expt,
        num_allies,
        marker
        ):

    device = torch_device(device=device)

    X, targets = load_processed_data(
        expt, 'processed_data_X_targets.pkl')
    log_shapes(
        [X] + [targets[i] for i in targets],
        locals(),
        'Dataset loaded'
    )

    targets = {i: elem.reshape(-1, 1) for i, elem in targets.items()}

    X_train, X_valid, \
        y_hef_train, y_hef_valid, \
        y_exf_train, y_exf_valid, \
        y_gdr_train, y_gdr_valid, \
        y_lan_train, y_lan_valid, \
        y_mar_train, y_mar_valid, \
        y_rel_train, y_rel_valid, \
        y_ins_train, y_ins_valid, \
        y_dis_train, y_dis_valid, \
        y_adl_train, y_adl_valid, \
        y_adt_train, y_adt_valid, \
        y_etn_train, y_etn_valid = train_test_split(
            X,
            targets['hospital_expire_flag'],
            targets['expire_flag'],
            targets['gender'],
            targets['language'],
            targets['marital_status'],
            targets['religion'],
            targets['insurance'],
            targets['discharge_location'],
            targets['admission_location'],
            targets['admission_type'],
            targets['ethnicity'],
            test_size=test_size,
            stratify=pd.DataFrame(np.concatenate(
                (
                    targets['admission_type'],
                ), axis=1)
            )
        )

    log_shapes(
        [
            X_train, X_valid,
            y_hef_train, y_hef_valid,
            y_exf_train, y_exf_valid,
            y_gdr_train, y_gdr_valid,
            y_lan_train, y_lan_valid,
            y_mar_train, y_mar_valid,
            y_rel_train, y_rel_valid,
            y_ins_train, y_ins_valid,
            y_dis_train, y_dis_valid,
            y_adl_train, y_adl_valid,
            y_adt_train, y_adt_valid,
            y_etn_train, y_etn_valid
        ],
        locals(),
        'Data size after train test split'
    )

    y_ally_trains = [
        y_hef_train,
        y_exf_train,
        y_gdr_train,
        y_lan_train,
        y_mar_train,
        y_rel_train,
        y_ins_train,
        y_dis_train,
        y_adl_train,
        y_etn_train,
    ]
    y_ally_valids = [
        y_hef_valid,
        y_exf_valid,
        y_gdr_valid,
        y_lan_valid,
        y_mar_valid,
        y_rel_valid,
        y_ins_valid,
        y_dis_valid,
        y_adl_valid,
        y_etn_valid,
    ]
    y_advr_train = y_adt_train
    y_advr_valid = y_adt_valid

    scaler = StandardScaler()
    X_normalized_train = scaler.fit_transform(X_train)
    X_normalized_valid = scaler.transform(X_valid)

    log_shapes([X_normalized_train, X_normalized_valid], locals())

    for i in [num_allies-1]:
        sep('NUMBER OF ALLIES: {}'.format(i+1))
        encoder = GeneratorFCN(
            X_normalized_train.shape[1], hidden_dim, encoding_dim,
            leaky, activation).to(device)
        ally = {}
        for j in range(i+1):
            ally[j] = DiscriminatorFCN(
                encoding_dim, hidden_dim, 1,
                leaky).to(device)
        advr = DiscriminatorFCN(
            encoding_dim, hidden_dim, 1,
            leaky).to(device)

        if init_weight:
            sep()
            logging.info('applying weights_init ...')
            encoder.apply(weights_init)
            for j in range(i+1):
                ally[j].apply(weights_init)
            advr.apply(weights_init)

        sep('encoder')
        summary(encoder, input_size=(1, X_normalized_train.shape[1]))
        for j in range(i+1):
            sep('ally:{}'.format(j))
            summary(ally[j], input_size=(1, encoding_dim))
        sep('advr')
        summary(advr, input_size=(1, encoding_dim))

        optim = torch.optim.Adam
        criterionBCEWithLogits = nn.BCEWithLogitsLoss()


        optimizer_encd = optim(encoder.parameters(), lr=lr_encd)
        optimizer_ally = {}
        for j in range(i+1):
            optimizer_ally[j] = optim(ally[j].parameters(), lr=lr_ally)
        optimizer_advr = optim(advr.parameters(), lr=lr_advr)

        dataset_train = utils.TensorDataset(
            torch.Tensor(X_normalized_train),
            torch.Tensor(y_advr_train),
        )
        for y_ally_train in y_ally_trains:
            dataset_train.tensors = (*dataset_train.tensors, torch.Tensor(y_ally_train))

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=1
        )

        dataset_valid = utils.TensorDataset(
            torch.Tensor(X_normalized_valid),
            torch.Tensor(y_advr_valid),
        )
        for y_ally_valid in y_ally_valids:
            dataset_valid.tensors = (*dataset_valid.tensors, torch.Tensor(y_ally_valid))

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
        ally_loss_train = {}
        ally_loss_valid = {}
        for j in range(i+1):
            ally_loss_train[j] = []
            ally_loss_valid[j] = []
        advr_loss_train = []
        advr_loss_valid = []

        log_list = ['epoch', 'encd_train', 'encd_valid', 'advr_train', 'advr_valid'] + \
            ['ally_{}_train \t ally_{}_valid'.format(str(j), str(j)) for j in range(i+1)]
        logging.info(' \t '.join(log_list))

        for epoch in range(n_epochs):

            encoder.train()
            for j in range(i+1):
                ally[i].eval()
            advr.eval()

            nsamples = 0
            iloss = 0
            for data in dataloader_train:
                X_train_torch = data[0].to(device)
                y_advr_train_torch = data[1].to(device)
                y_ally_train_torch = {}
                for j in range(i+1):
                    y_ally_train_torch[j] = data[j+2].to(device)

                optimizer_encd.zero_grad()
                # Forward pass
                X_train_encoded = encoder(X_train_torch)
                y_advr_train_hat_torch = advr(X_train_encoded)
                y_ally_train_hat_torch = {}
                for j in range(i+1):
                    y_ally_train_hat_torch[j] = ally[j](X_train_encoded)
                # Compute Loss
                loss_ally = {}
                for j in range(i+1):
                    loss_ally[j] = criterionBCEWithLogits(
                        y_ally_train_hat_torch[j], y_ally_train_torch[j])
                loss_advr = criterionBCEWithLogits(
                    y_advr_train_hat_torch,
                    y_advr_train_torch)

                loss_encd = alpha/num_allies * sum([loss_ally[_].item() for _ in loss_ally]) - (1-alpha) * loss_advr
                # Backward pass
                loss_encd.backward()
                optimizer_encd.step()

                nsamples += 1
                iloss += loss_encd.item()

            epochs_train.append(epoch)
            encd_loss_train.append(iloss/nsamples)

            encoder.eval()
            for j in range(i+1):
                ally[j].train()
            advr.train()

            nsamples = 0
            iloss_ally = {}
            for j in range(i+1):
                iloss_ally[j] = 0
            iloss_advr = 0
            for data in dataloader_train:
                X_train_torch = data[0].to(device)
                y_advr_train_torch = data[1].to(device)
                y_ally_train_torch = {}
                for j in range(i+1):
                    y_ally_train_torch[j] = data[j+2].to(device)

                y_ally_train_hat_torch = {}
                loss_ally = {}
                for j in range(i+1):
                    optimizer_ally[j].zero_grad()
                    X_train_encoded = encoder(X_train_torch)
                    y_ally_train_hat_torch[j] = ally[j](X_train_encoded)
                    loss_ally[j] = criterionBCEWithLogits(
                        y_ally_train_hat_torch[j], y_ally_train_torch[j])
                    loss_ally[j].backward()
                    optimizer_ally[j].step()

                optimizer_advr.zero_grad()
                X_train_encoded = encoder(X_train_torch)
                y_advr_train_hat_torch = advr(X_train_encoded)
                loss_advr = criterionBCEWithLogits(
                    y_advr_train_hat_torch,
                    y_advr_train_torch)
                loss_advr.backward()
                optimizer_advr.step()

                nsamples += 1
                for j in range(i+1):
                    iloss_ally[j] += loss_ally[j].item()
                iloss_advr += loss_advr.item()

            for j in range(i+1):
                ally_loss_train[j].append(iloss_ally[j]/nsamples)
            advr_loss_train.append(iloss_advr/nsamples)

            if epoch % int(n_epochs/10) != 0:
                continue

            encoder.eval()
            for j in range(i+1):
                ally[j].eval()
            advr.eval()

            nsamples = 0
            iloss = 0
            iloss_ally = {}
            for j in range(i+1):
                iloss_ally[j] = 0
            iloss_advr = 0

            for data in dataloader_valid:
                X_valid_torch = data[0].to(device)
                y_advr_valid_torch = data[1].to(device)
                y_ally_valid_torch = {}
                for j in range(i+1):
                    y_ally_valid_torch[j] = data[j+2].to(device)

                X_valid_encoded = encoder(X_valid_torch)
                y_ally_valid_hat_torch = {}
                for j in  range(i+1):
                    y_ally_valid_hat_torch[j] = ally[j](X_valid_encoded)
                y_advr_valid_hat_torch = advr(X_valid_encoded)

                valid_loss_ally = {}
                for j in range(i+1):
                    valid_loss_ally[j] = criterionBCEWithLogits(
                        y_ally_valid_hat_torch[j], y_ally_valid_torch[j])
                valid_loss_advr = criterionBCEWithLogits(
                    y_advr_valid_hat_torch, y_advr_valid_torch)
                valid_loss_encd = alpha/num_allies*sum(
                    [valid_loss_ally[_].item() for _ in valid_loss_ally]
                ) - (1-alpha)* valid_loss_advr

                nsamples += 1
                iloss += valid_loss_encd.item()
                for j in range(i+1):
                    iloss_ally[j] += valid_loss_ally[j].item()
                iloss_advr += valid_loss_advr.item()

            epochs_valid.append(epoch)
            encd_loss_valid.append(iloss/nsamples)
            for j in range(i+1):
                ally_loss_valid[j].append(iloss_ally[j]/nsamples)
            advr_loss_valid.append(iloss_advr/nsamples)

            log_line = [str(epoch), '{:.8f}'.format(encd_loss_train[-1]), '{:.8f}'.format(encd_loss_valid[-1]),
                '{:.8f}'.format(advr_loss_train[-1]), '{:.8f}'.format(advr_loss_valid[-1]),
            ] + \
            [
                '{:.8f} \t {:.8f}'.format(
                    ally_loss_train[_][-1],
                    ally_loss_valid[_][-1]
                ) for _ in ally_loss_train]
            logging.info(' \t '.join(log_line))

        config_summary = '{}_n_{}_device_{}_dim_{}_hidden_{}_batch_{}_epochs_{}_ally_{}_encd_{:.4f}_advr_{:.4f}'\
            .format(
                marker,
                num_allies,
                device,
                encoding_dim,
                hidden_dim,
                batch_size,
                n_epochs,
                i,
                encd_loss_train[-1],
                advr_loss_valid[-1],
            )

        plt.figure()
        plt.plot(epochs_train, encd_loss_train, 'r', label='encd train')
        plt.plot(epochs_valid, encd_loss_valid, 'r--', label='encd valid')
        # sum_loss = [0] * len(ally_loss_train[0])
        # for j in range(i+1):
        #     for k in ally_loss_valid:
        #         sum_loss[j] += ally_loss_train[j]
        # sum_loss = [sum_loss[j]/len(ally_loss_valid) for j in sum_loss]
        # plt.plot(
        #     epochs_train,
        #     sum([ally_loss_train[j] for j in range(i+1)])/len(ally_loss_train),
        #     'b', label='ally_sum_train')
        # sum_loss = [0] * len(ally_loss_valid)
        # for j in range(i+1):
        #     sum_loss[j] += ally_loss_valid[j]
        # sum_loss = [sum_loss[j]/len(ally_loss_valid) for j in sum_loss]
        # plt.plot(epochs_valid, sum_loss, 'b', label='ally_sum_valid')
        plt.plot(epochs_train, advr_loss_train, 'g', label='advr_train')
        plt.plot(epochs_valid, advr_loss_valid, 'g--', label='advr_valid')
        plt.legend()
        plt.title("{} on {} training".format(model, expt))

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
            advr_loss_train, advr_loss_valid,
        ), open(checkpoint_location, 'wb'))

        model_ckpt = 'checkpoints/{}/{}_torch_model_{}_{}.pkl'.format(
                expt, model, time_stamp, config_summary)
        logging.info('Saving: {}'.format(model_ckpt))
        torch.save(encoder, model_ckpt)


if __name__ == "__main__":
    expt = 'mimic'
    model = 'n_eigan'
    marker = 'B'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    args = eigan_argparse()
    main(
        model=model,
        time_stamp=fl_time,
        device=args['device'],
        ngpu=args['n_gpu'],
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
        lr_advr=args['lr_advr'],
        alpha=args['alpha'],
        expt=args['expt'],
        marker=marker,
        num_allies=args['num_allies'],
    )
    log_time('End', time_stp()[0])
    sep()
