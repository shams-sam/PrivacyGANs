import joblib
import logging
import matplotlib
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

matplotlib.rcParams.update({'font.size': 12})

def main(expt, model):
    pca_1 = pkl.load(open('checkpoints/titanic/ind_pca_training_history_01_15_2020_23_25_44.pkl', 'rb'))
    pca_2 = pkl.load(open('checkpoints/titanic/ind_pca_training_history_01_15_2020_23_45_00.pkl', 'rb'))
    auto_1 = pkl.load(open('checkpoints/titanic/ind_autoencoder_training_history_01_16_2020_03_53_53.pkl', 'rb'))
    auto_2 = pkl.load(open('checkpoints/titanic/ind_autoencoder_training_history_01_16_2020_04_30_49.pkl', 'rb'))
    dp_1 = pkl.load(open('checkpoints/titanic/ind_dp_training_history_01_30_2020_14_11_06.pkl', 'rb'))
    gan_1 = pkl.load(open('checkpoints/titanic/ind_gan_training_history_01_16_2020_21_56_04.pkl', 'rb'))

    s = pkl.load(open('checkpoints/titanic/eigan_training_history_01_15_2020_22_43_42_E_device_cuda_dim_1400_hidden_2800_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-0.1852_val_0.6462.pkl', 'rb'))
    # checkpoints/titanic/ind_gan_training_history_01_16_2020_21_56_04.pkl

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
    ], prop={'size': 10})
    ax3.set_title(t3, y=-0.32)
    ax3.set_xlabel('epochs')
    ax3.set_ylabel('log loss')
    ax3.grid()
    ax3.set_xlim(left=0, right=1000)
    ax3.text(320,0.67, 'Lower is better', fontsize=12, color='r')

    ax1.plot(pca_1['epoch']['valid'], gan_1['encoder']['advr_1_valid'], 'r', label='EIGAN adversary')
    ax1.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['advr_1_valid'], 'b', label='Autoencoder adversary')
    ax1.plot(pca_1['epoch']['valid'], pca_1['pca']['advr_1_valid'], 'g', label='PCA adversary')
    ax1.plot(pca_1['epoch']['valid'], dp_1['dp']['advr_1_valid'], 'y', label='DP adversary')
    ax1.plot(pca_1['epoch']['valid'], gan_1['encoder']['advr_2_valid'], 'r--')
    ax1.plot(pca_1['epoch']['valid'], auto_2['autoencoder']['advr_2_valid'], 'b--')
    ax1.plot(pca_1['epoch']['valid'], pca_2['pca']['advr_2_valid'], 'g--')
    ax1.plot(pca_1['epoch']['valid'], dp_1['dp']['advr_2_valid'], 'y--')
    ax1.legend(prop={'size': 10})
    ax1.set_title(t1, y=-0.32)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('log loss')
    ax1.grid()
    ax1.set_xlim(left=0, right=1000)
    ax1.text(320, 0.66, 'Higher is better', fontsize=12, color='r')

    ax2.plot(s[0], s[2], 'r', label='encoder loss')
    ax2.set_title('(c)', y=-0.32)
    ax2.plot(np.nan, 'b', label = 'adversary loss')
    ax2.legend(prop={'size':10})
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('encoder loss')
    ax2.grid()
    ax2.set_xlim(left=0, right=1000)
    ax4 = ax2.twinx()
    ax4.plot(s[0], s[6], 'b')
    ax4.set_ylabel('adversary loss')

    fig.subplots_adjust(wspace=0.3)

    plot_location = 'plots/{}/{}_{}_.png'.format(expt, 'all', model)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location, bbox_inches='tight', dpi=300)


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
