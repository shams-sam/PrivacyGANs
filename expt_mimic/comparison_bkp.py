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
    pca_1 = pkl.load(open('checkpoints/mimic/ind_pca_training_history_01_20_2020_23_31_01.pkl', 'rb'))
    pca_2 = pkl.load(open('checkpoints/mimic/ind_pca_training_history_01_21_2020_00_19_41.pkl', 'rb'))
    auto_1 = pkl.load(open('checkpoints/mimic/ind_autoencoder_training_history_01_24_2020_13_50_25.pkl', 'rb'))
    dp_1 = pkl.load(open('checkpoints/mimic/ind_dp_training_history_01_24_2020_07_31_44.pkl', 'rb'))
    gan_1 = pkl.load(open('checkpoints/mimic/ind_gan_training_history_01_24_2020_13_16_16.pkl', 'rb'))
    gan_2 = pkl.load(open('checkpoints/mimic/ind_gan_training_history_01_25_2020_02_04_34.pkl', 'rb'))
    # gan_1 = pkl.load(open('checkpoints/mimic/ind_gan_training_history_01_27_2020_00_57_38.pkl', 'rb'))
    # gan_2 = gan_1

    # print(pca_1.keys(), pca_2.keys(), auto_1.keys(), auto_2.keys(), dp_1.keys(), gan_1.keys())
    # return
    plt.figure()
    fig = plt.figure(figsize=(15, 3))
    ax3 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    t3, t1, t2 = '(a)', '(b)', '(c)'

    ax3.plot(pca_1['epoch']['valid'], gan_1['encoder']['ally_valid'], 'r')
    ax3.plot(pca_1['epoch']['valid'], pca_1['pca']['ally_valid'], 'g')
    ax3.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['ally_valid'], 'b')
    ax3.plot(pca_1['epoch']['valid'], dp_1['dp']['ally_valid'], 'y')
    ax3.legend([
        'EIGAN ally',
        'Autoencoder ally',
        'PCA ally',
        'DP ally',
    ])
    ax3.set_title(t3, y=-0.3)
    ax3.set_xlabel('epochs')
    ax3.set_ylabel('log loss')
    ax3.grid()
    ax3.text(320,0.618, 'Lower is better', fontsize=12, color='r')
    ax3.set_ylim(bottom=0.58)

    ax1.plot(pca_1['epoch']['valid'], gan_2['encoder']['advr_1_valid'], 'r--')
    ax1.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['advr_1_valid'], 'b--')
    ax1.plot(pca_1['epoch']['valid'], pca_1['pca']['advr_1_valid'], 'g--')
    ax1.plot(pca_1['epoch']['valid'], dp_1['dp']['advr_1_valid'], 'y--')
    ax1.legend([
        'EIGAN adversary 1',
        'Autoencoder adversary 1',
        'PCA adversary 1',
        'DP adversary 1',
    ])
    ax1.set_title(t1, y=-0.3)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('log loss')
    ax1.grid()
    ax1.text(320,0.67, 'Higher is better', fontsize=12, color='r')
    ax1.set_ylim(bottom=0.66)

    ax2.plot(pca_1['epoch']['valid'], gan_1['encoder']['advr_2_valid'], 'r--')
    ax2.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['advr_2_valid'], 'b--')
    ax2.plot(pca_1['epoch']['valid'], pca_2['pca']['advr_2_valid'], 'g--')
    ax2.plot(pca_1['epoch']['valid'], dp_1['dp']['advr_2_valid'], 'y--')
    ax2.legend([
        'EIGAN adversary 2',
        'Autoencoder adversary 2',
        'PCA adversary 2',
        'DP adversary 2',
    ])
    ax2.set_title(t2, y=-0.3)
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('log loss')
    ax2.grid()
    ax2.text(320,0.56, 'Higher is better', fontsize=12, color='r')
    ax2.set_ylim(bottom=0.54, top=0.64)

    fig.subplots_adjust(wspace=0.3)

    plot_location = 'plots/{}/{}_{}.png'.format(expt, 'all', model)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location, bbox_inches='tight')


if __name__ == "__main__":
    expt = 'mimic'
    model = 'comparison'
    marker = 'A'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    main(expt, model)
    log_time('End', time_stp()[0])
    sep()
