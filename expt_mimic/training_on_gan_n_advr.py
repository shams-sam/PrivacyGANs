import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import logging
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.utils.data as utils
from models.eigan import DiscriminatorFCN
from common.torchsummary import summary
from common.utility import log_shapes, log_time, torch_device,\
    time_stp, logger, sep, weights_init, load_processed_data
from common.argparser import comparison_argparse


def main(
        model,
        time_stamp,
        device,
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

    targets = {i: elem.reshape(-1, 1) for i, elem in targets.items()}

    X_train, X_valid, \
        y_adt_train, y_adt_valid = train_test_split(
            X,
            targets['admission_type'],
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
            y_adt_train, y_adt_valid,
        ],
        locals(),
        'Data size after train test split'
    )

    y_train = y_adt_train
    y_valid = y_adt_valid

    scaler = StandardScaler()
    X_normalized_train = scaler.fit_transform(X_train)
    X_normalized_valid = scaler.transform(X_valid)

    log_shapes([X_normalized_train, X_normalized_valid], locals())

    ckpts = {
        # 12A: checkpoints/mimic/n_advr_ind_gan_training_history_02_12_2020_19_30_38.pkl
        # 0: 'checkpoints/mimic/n_advr_eigan_torch_model_02_12_2020_00_10_16_A_n_1_device_cuda_dim_256_hidden_512_batch_32768_epochs_1001_advr_0_encd_-0.1220_advr_0.4970.pkl',
        # 12A: checkpoints/mimic/n_advr_ind_gan_training_history_02_12_2020_19_30_38.pkl
        # 1: 'checkpoints/mimic/n_advr_eigan_torch_model_02_12_2020_02_02_15_A_n_2_device_cuda_dim_256_hidden_512_batch_32768_epochs_1001_advr_1_encd_-0.1086_advr_0.4969.pkl',
        # 34A: checkpoints/mimic/n_advr_ind_gan_training_history_02_12_2020_20_01_01.pkl
        # 2: 'checkpoints/mimic/n_advr_eigan_torch_model_02_12_2020_15_03_17_A_n_3_device_cuda_dim_256_hidden_512_batch_32768_epochs_1001_advr_2_encd_-0.0949_advr_0.4908.pkl',
        # 34A: checkpoints/mimic/n_advr_ind_gan_training_history_02_12_2020_20_01_01.pkl
        # 3: 'checkpoints/mimic/n_advr_eigan_torch_model_02_12_2020_15_02_44_A_n_4_device_cuda_dim_256_hidden_512_batch_32768_epochs_1001_advr_3_encd_-0.1082_advr_0.4981.pkl',
        # 56A: checkpoints/mimic/n_advr_ind_gan_training_history_02_12_2020_23_07_22.pkl
        # 4: 'checkpoints/mimic/n_advr_eigan_torch_model_02_12_2020_20_01_18_A_n_5_device_cuda_dim_256_hidden_512_batch_32768_epochs_1001_advr_4_encd_-0.1145_advr_0.4966.pkl',
        # 56A: checkpoints/mimic/n_advr_ind_gan_training_history_02_12_2020_23_07_22.pkl
        # 5: 'checkpoints/mimic/n_advr_eigan_torch_model_02_12_2020_21_34_32_A_n_6_device_cuda_dim_256_hidden_512_batch_32768_epochs_1001_advr_5_encd_-0.1172_advr_0.4960.pkl',
        # 789A: checkpoints/mimic/n_advr_ind_gan_training_history_02_13_2020_13_54_53.pkl
        # 6: 'checkpoints/mimic/n_advr_eigan_torch_model_02_12_2020_23_56_04_A_n_7_device_cuda_dim_256_hidden_512_batch_32768_epochs_1001_advr_6_encd_-0.0955_advr_0.4904.pkl',
        # 789A: checkpoints/mimic/n_advr_ind_gan_training_history_02_13_2020_13_54_53.pkl
        # 7: 'checkpoints/mimic/n_advr_eigan_torch_model_02_12_2020_23_59_23_A_n_8_device_cuda_dim_256_hidden_512_batch_16384_epochs_1001_advr_7_encd_-0.1173_advr_0.4964.pkl',
        # 789A: checkpoints/mimic/n_advr_ind_gan_training_history_02_13_2020_13_54_53.pkl
        # 8: 'checkpoints/mimic/n_advr_eigan_torch_model_02_13_2020_04_35_19_A_n_9_device_cuda_dim_256_hidden_512_batch_32768_epochs_1001_advr_8_encd_-0.1105_advr_0.4947.pkl',
        # nA: checkpoints/mimic/n_advr_ind_gan_training_history_02_13_2020_18_31_15.pkl
        # 9: 'checkpoints/mimic/n_advr_eigan_torch_model_02_13_2020_13_32_21_A_n_10_device_cuda_dim_256_hidden_512_batch_32768_epochs_1001_advr_9_encd_-0.1104_advr_0.4955.pkl',
    }

    h = {}

    for idx, ckpt in ckpts.items():
        encoder = torch.load(ckpt, map_location=device)
        encoder.eval()

        optim = torch.optim.Adam
        criterionBCEWithLogits = nn.BCEWithLogitsLoss()

        h[idx] = {
            'epoch_train': [],
            'epoch_valid': [],
            'advr_train': [],
            'advr_valid': [],
        }

        dataset_train = utils.TensorDataset(
            torch.Tensor(X_normalized_train),
            torch.Tensor(y_train),
        )

        dataset_valid = utils.TensorDataset(
            torch.Tensor(X_normalized_valid),
            torch.Tensor(y_valid),
        )

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

        clf = DiscriminatorFCN(
            encoding_dim, hidden_dim, 1,
            leaky).to(device)

        clf.apply(weights_init)

        sep('{} {}'.format(idx+1, 'ally'))
        summary(clf, input_size=(1, encoding_dim))

        optimizer = optim(clf.parameters(), lr=lr)

        # adversary 1
        sep("adversary with {} ally encoder".format(idx+1))
        logging.info('{} \t {} \t {}'.format(
            'Epoch',
            'Advr Train',
            'Advr Valid',
        ))

        for epoch in range(n_epochs):
            clf.train()

            nsamples = 0
            iloss_advr = 0
            for i, data in enumerate(dataloader_train, 0):
                X_train_torch = transform(data[0].to(device))
                y_advr_train_torch = data[1].to(device)

                optimizer.zero_grad()
                y_advr_train_hat_torch = clf(X_train_torch)

                loss_advr = criterionBCEWithLogits(
                    y_advr_train_hat_torch, y_advr_train_torch)
                loss_advr.backward()
                optimizer.step()

                nsamples += 1
                iloss_advr += loss_advr.item()

            h[idx]['advr_train'].append(iloss_advr/nsamples)
            h[idx]['epoch_train'].append(epoch)

            if epoch % int(n_epochs/10) != 0:
                continue

            clf.eval()

            nsamples = 0
            iloss_advr = 0
            correct = 0
            total = 0

            for i, data in enumerate(dataloader_valid, 0):
                X_valid_torch = transform(data[0].to(device))
                y_advr_valid_torch = data[1].to(device)
                y_advr_valid_hat_torch = clf(X_valid_torch)

                valid_loss_advr = criterionBCEWithLogits(
                    y_advr_valid_hat_torch, y_advr_valid_torch,)

                predicted = y_advr_valid_hat_torch > 0.5

                nsamples += 1
                iloss_advr += valid_loss_advr.item()
                total += y_advr_valid_torch.size(0)
                correct += (predicted == y_advr_valid_torch).sum().item()

            h[idx]['advr_valid'].append(iloss_advr/nsamples)
            h[idx]['epoch_valid'].append(epoch)

            logging.info(
                '{} \t {:.8f} \t {:.8f} \t {:.8f}'.
                format(
                    epoch,
                    h[idx]['advr_train'][-1],
                    h[idx]['advr_valid'][-1],
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
    model = 'n_advr_ind_gan'
    marker = 'nA'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    args = comparison_argparse()
    main(
        model=model,
        time_stamp=fl_time,
        device=args['device'],
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
