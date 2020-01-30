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


def main(expt, model):
    gan_d_128 = pkl.load(open('checkpoints/titanic/eigan_training_history_01_25_2020_22_26_59_F_device_cuda_dim_128_hidden_256_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-0.1927_val_0.6559.pkl', 'rb'))
    gan_d_256 = pkl.load(open('checkpoints/titanic/eigan_training_history_01_25_2020_22_29_45_F_device_cuda_dim_256_hidden_512_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-0.1852_val_0.6548.pkl', 'rb'))
    gan_d_512 = pkl.load(open('checkpoints/titanic/eigan_training_history_01_25_2020_22_32_52_F_device_cuda_dim_512_hidden_1024_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-0.1820_val_0.6553.pkl', 'rb'))
    gan_d_1024 = pkl.load(open('checkpoints/titanic/eigan_training_history_01_25_2020_22_36_17_F_device_cuda_dim_1024_hidden_2048_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-0.1834_val_0.6484.pkl', 'rb'))
    gan_d_2048 = pkl.load(open('checkpoints/titanic/eigan_training_history_01_25_2020_22_40_32_F_device_cuda_dim_2048_hidden_4086_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-0.1826_val_0.6424.pkl', 'rb'))

    # print(pca_1.keys(), pca_2.keys(), auto_1.keys(), auto_2.keys(), dp_1.keys(), gan_1.keys())
    # return
    plt.figure()
    fig = plt.figure(figsize=(15,5))
    ax3 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    t3, t1, t2 = '(a)', '(b)', '(c)'

    ax3.plot(pca_1['epoch']['valid'], gan_1['encoder']['ally_valid'], 'r')
    ax3.plot(pca_1['epoch']['valid'], pca_1['pca']['ally_valid'], 'g')
    ax3.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['ally_valid'], 'b')
    ax3.plot(pca_1['epoch']['valid'], dp_1['dp']['ally_valid'], 'y')
    ax3.legend([
        'gan ally',
        'autoencoder ally',
        'pca ally',
        'dp ally',
    ])
    ax3.set_title(t3, y=-0.2)
    ax3.set_xlabel('iterations (scale adjusted)')
    ax3.set_ylabel('loss')

    ax1.plot(pca_1['epoch']['valid'], gan_1['encoder']['advr_1_valid'], 'r--')
    ax1.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['advr_1_valid'], 'b--')
    ax1.plot(pca_1['epoch']['valid'], pca_1['pca']['advr_1_valid'], 'g--')
    ax1.plot(pca_1['epoch']['valid'], dp_1['dp']['advr_1_valid'], 'y--')
    ax1.legend([
        'gan adversary 1',
        'autoencoder adversary 1',
        'pca adversary 1',
        'dp adversary 1',
    ])
    ax1.set_title(t1, y=-0.2)
    ax1.set_xlabel('iterations (scale adjusted)')
    ax1.set_ylabel('loss')

    ax2.plot(pca_1['epoch']['valid'], gan_1['encoder']['advr_2_valid'], 'r--')
    ax2.plot(pca_1['epoch']['valid'], auto_2['autoencoder']['advr_2_valid'], 'b--')
    ax2.plot(pca_1['epoch']['valid'], pca_2['pca']['advr_2_valid'], 'g--')
    ax2.plot(pca_1['epoch']['valid'], dp_1['dp']['advr_2_valid'], 'y--')
    ax2.legend([
        'gan adversary 2',
        'autoencoder adversary 2',
        'pca adversary 2',
        'dp adversary 2',
    ])
    ax2.set_title(t2, y=-0.2)
    ax2.set_xlabel('iterations (scale adjusted)')
    ax2.set_ylabel('loss')

    plot_location = 'plots/{}/{}_{}_b4096.png'.format(expt, 'all', model)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location, bbox_inches='tight')


if __name__ == "__main__":
    expt = 'titanic'
    model = 'comparison'
    marker = 'A'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    main(expt, model)
    log_time('End', time_stp()[0])
    sep()
