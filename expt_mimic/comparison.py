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

import matplotlib

matplotlib.rcParams.update({'font.size': 12})



def main(expt, model):
    pca_1 = pkl.load(open('checkpoints/mimic_centralized/ind_pca_training_history_01_20_2020_23_31_01.pkl', 'rb'))
    pca_2 = pkl.load(open('checkpoints/mimic_centralized/ind_pca_training_history_01_21_2020_00_19_41.pkl', 'rb'))
    auto_1 = pkl.load(open('checkpoints/mimic_centralized/ind_autoencoder_training_history_01_24_2020_13_50_25.pkl', 'rb'))
    dp_1 = pkl.load(open('checkpoints/mimic_centralized/ind_dp_training_history_01_24_2020_07_31_44.pkl', 'rb'))
    gan_1 = pkl.load(open('checkpoints/mimic_centralized/ind_gan_training_history_01_24_2020_13_16_16.pkl', 'rb'))
    gan_2 = pkl.load(open('checkpoints/mimic_centralized/ind_gan_training_history_01_25_2020_02_04_34.pkl', 'rb'))
    st_1_2 = pkl.load(open('checkpoints/mimic/ind_gan_dist_s1_2_training_history_04_07_2020_15_44_42.pkl', 'rb'))
    st_1_3 = pkl.load(open('checkpoints/mimic/ind_gan_dist_s1_nodes_3_training_history_04_16_2020_23_06_10.pkl', 'rb'))
    st_1_4 = pkl.load(open('checkpoints/mimic/ind_gan_dist_s1_nodes_4_training_history_04_17_2020_00_29_50.pkl', 'rb'))
    st_1_6 = pkl.load(open('checkpoints/mimic/ind_gan_dist_s1_nodes_6_training_history_04_17_2020_01_23_53.pkl', 'rb'))
    st_1_8 = pkl.load(open('checkpoints/mimic/ind_gan_dist_s1_nodes_8_training_history_04_17_2020_12_58_18.pkl', 'rb'))
    st_1_12 = pkl.load(open('checkpoints/mimic/ind_gan_dist_s1_nodes_12_training_history_04_17_2020_14_40_30.pkl', 'rb'))
    # gan_1 = pkl.load(open('checkpoints/mimic/ind_gan_training_history_01_27_2020_00_57_38.pkl', 'rb'))
    # gan_2 = gan_1

    s = pkl.load(open('checkpoints/mimic_centralized/n_eigan_training_history_02_03_2020_00_59_27_B_device_cuda_dim_256_hidden_512_batch_16384_epochs_1001_ally_0_encd_0.0276_advr_0.5939.pkl','rb'))

    # print(pca_1.keys(), pca_2.keys(), auto_1.keys(), auto_2.keys(), dp_1.keys(), gan_1.keys())
    # return
    plt.figure()
    fig = plt.figure(figsize=(15, 6))
    ax3 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    t3, t1, t2 = '(a)', '(b)', '(c)'

    ax3.plot(pca_1['epoch']['valid'], gan_1['encoder']['ally_valid'], 'r')
    ax3.plot(pca_1['epoch']['valid'], st_1_2['encoder']['ally_valid'], 'k*')
    ax3.plot(pca_1['epoch']['valid'], st_1_3['encoder']['ally_valid'], 'k--')
    ax3.plot(pca_1['epoch']['valid'], st_1_4['encoder']['ally_valid'], 'k+')
    ax3.plot(pca_1['epoch']['valid'], st_1_6['encoder']['ally_valid'], 'ks')
    ax3.plot(pca_1['epoch']['valid'], st_1_8['encoder']['ally_valid'], 'k.')
    ax3.plot(pca_1['epoch']['valid'], st_1_12['encoder']['ally_valid'], 'k')
    ax3.plot(pca_1['epoch']['valid'], pca_1['pca']['ally_valid'], 'g')
    ax3.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['ally_valid'], 'b')
    ax3.plot(pca_1['epoch']['valid'], dp_1['dp']['ally_valid'], 'y')
    ax3.legend([
        'EIGAN ally',
        '2 nodes',
        '3 nodes',
        '4 nodes',
        '6 nodes',
        '8 nodes',
        '12 nodes',
        'Autoencoder',
        'PCA',
        'DP',
    ],prop={'size':10})
    ax3.set_title(t3, y=-0.32)
    ax3.set_xlabel('epochs')
    ax3.set_ylabel('log loss')
    ax3.grid()
    # ax3.text(320,0.618, 'Lower is better', fontsize=12, color='r')
    # ax3.set_ylim(bottom=0.58, top=0.8)
    # ax3.set_xlim(left=0, right=1000)

    ax1.plot(pca_1['epoch']['valid'], gan_2['encoder']['advr_1_valid'], 'r', label='EIGAN adversary')
    ax1.plot(pca_1['epoch']['valid'], st_1_2['encoder']['advr_1_valid'], 'k*', label='2 nodes')
    ax1.plot(pca_1['epoch']['valid'], st_1_3['encoder']['advr_1_valid'], 'k--', label='3 nodes')
    ax1.plot(pca_1['epoch']['valid'], st_1_4['encoder']['advr_1_valid'], 'k+', label='4 nodes')
    ax1.plot(pca_1['epoch']['valid'], st_1_6['encoder']['advr_1_valid'], 'ks', label='6 nodes')
    ax1.plot(pca_1['epoch']['valid'], st_1_8['encoder']['advr_1_valid'], 'k.', label='8 nodes')
    ax1.plot(pca_1['epoch']['valid'], st_1_12['encoder']['advr_1_valid'], 'k', label='12 nodes')
    ax1.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['advr_1_valid'], 'b', label='Autoencoder')
    ax1.plot(pca_1['epoch']['valid'], pca_1['pca']['advr_1_valid'], 'g', label='PCA')
    ax1.plot(pca_1['epoch']['valid'], dp_1['dp']['advr_1_valid'], 'y', label='DP')
    ax1.legend(prop={'size':10})
    ax1.set_title(t1, y=-0.32)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('log loss')
    ax1.grid()
    # ax1.text(320,0.58, 'Higher is better', fontsize=12, color='r')
    # ax1.set_ylim(bottom=0.53, top=0.81)
    # ax1.set_xlim(left=0, right=1000)


    ax2.plot(pca_1['epoch']['valid'], gan_1['encoder']['advr_2_valid'], 'r', label='EIGAN Adversary 2')
    ax2.plot(pca_1['epoch']['valid'], st_1_2['encoder']['advr_2_valid'], 'k*', label='2 nodes')
    ax2.plot(pca_1['epoch']['valid'], st_1_3['encoder']['advr_2_valid'], 'k--', label='3 nodes')
    ax2.plot(pca_1['epoch']['valid'], st_1_4['encoder']['advr_2_valid'], 'k+', label='4 nodes')
    ax2.plot(pca_1['epoch']['valid'], st_1_6['encoder']['advr_2_valid'], 'ks', label='6 nodes')
    ax2.plot(pca_1['epoch']['valid'], st_1_8['encoder']['advr_2_valid'], 'k.', label='8 nodes')
    ax2.plot(pca_1['epoch']['valid'], st_1_12['encoder']['advr_2_valid'], 'k', label='12 nodes')
    ax2.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['advr_2_valid'], 'b', label='autoencoder')
    ax2.plot(pca_1['epoch']['valid'], pca_2['pca']['advr_2_valid'], 'g', label='PCA')
    ax2.plot(pca_1['epoch']['valid'], dp_1['dp']['advr_2_valid'], 'y', label='DP')
    # ax2.plot(s[0], s[2], 'r', label='encoder loss')
    ax2.set_title('(c)', y=-0.32)
    # ax2.plot(np.nan, 'b', label = 'adversary loss')
    ax2.legend(prop={'size':10})
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('log loss')
    ax2.grid()
    # ax2.set_xlim(left=0,right=500)
    # ax3 = ax2.twinx()
    # ax3.plot(s[0], s[6], 'b')
    # ax3.set_ylabel('adversary loss')

    fig.subplots_adjust(wspace=0.3)

    plot_location = 'plots/{}/{}_{}.png'.format(expt, 'all', model)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location, bbox_inches='tight', dpi=300)


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
