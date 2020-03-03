import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib

matplotlib.rcParams.update({'font.size': 12})

def ys(y, sigma=4):
    if sigma==0:
        return y
    return gaussian_filter1d(y, sigma=sigma)


h = {}

h[4] = pkl.load(open('checkpoints/mimic/eigan_training_history_01_31_2020_23_50_47_F1_device_cuda_dim_4_hidden_8_batch_16384_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.6608_val_0.6893.pkl', 'rb'))
h[8] = pkl.load(open('checkpoints/mimic/eigan_training_history_02_01_2020_00_17_12_F1_device_cuda_dim_8_hidden_16_batch_16384_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.6749_val_0.6825.pkl', 'rb'))
h[16] = pkl.load(open('checkpoints/mimic/eigan_training_history_02_01_2020_00_43_58_F1_device_cuda_dim_16_hidden_32_batch_16384_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.6853_val_0.6866.pkl', 'rb'))
h[32] = pkl.load(open('checkpoints/mimic/eigan_training_history_02_01_2020_01_10_58_F1_device_cuda_dim_32_hidden_64_batch_16384_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.6956_val_0.6883.pkl', 'rb'))
h[64] = pkl.load(open('checkpoints/mimic/eigan_training_history_02_01_2020_01_43_39_F1_device_cuda_dim_64_hidden_128_batch_16384_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.7060_val_0.6864.pkl', 'rb'))
h[128] = pkl.load(open('checkpoints/mimic/eigan_training_history_01_26_2020_02_40_01_F_device_cuda_dim_128_hidden_256_batch_1024_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.7183_val_0.6851.pkl', 'rb'))
h[256] = pkl.load(open('checkpoints/mimic/eigan_training_history_01_26_2020_03_03_43_F_device_cuda_dim_256_hidden_512_batch_1024_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.7165_val_0.6856.pkl', 'rb'))
h[512] = pkl.load(open('checkpoints/mimic/eigan_training_history_02_05_2020_18_13_02_F1_device_cuda_dim_512_hidden_1024_batch_16384_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.6998_val_0.6870.pkl', 'rb'))
h[1024] = pkl.load(open('checkpoints/mimic/eigan_training_history_01_26_2020_04_00_57_F_device_cuda_dim_1024_hidden_2048_batch_1024_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.7371_val_0.6840.pkl', 'rb'))
h[2048] = pkl.load(open('checkpoints/mimic/eigan_training_history_01_26_2020_05_12_44_F_device_cuda_dim_2048_hidden_4096_batch_1024_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.7425_val_0.6931.pkl', 'rb'))
# h[4096] = pkl.load(open('checkpoints/mimic/eigan_training_history_02_05_2020_18_42_02_F1_device_cuda_dim_4096_hidden_8192_batch_4096_epochs_1001_lrencd_0.001_lrally_0.0001_tr_-0.6819_val_0.7558.pkl', 'rb'))


sigma = 2

threshold = []
for batch, hist in h.items():
    threshold.append(0.99*np.mean(hist[2][-100:]))

converge = []
for batch, hist in h.items():
    for idx, _ in enumerate(hist[2]):
        if _ < threshold[int(np.log(batch)-2)]:
            converge.append(idx+5)
            break

minimum = []
for batch, hist in h.items():
    minimum.append(min(ys(hist[2], sigma)))

batch = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 
# 4096
]

fig = plt.figure(figsize=(5, 3))
ax1 = fig.add_subplot(111)

ax2 = ax1.twinx()
ax1.scatter(np.nan, np.nan, c='b', marker='s', label = 'final loss on convergence')
ax1.plot(batch, ys(converge, sigma=sigma), 'r--')
ax1.scatter(batch, converge, c='r', marker='o', label='#epochs until convergence')
ax2.plot(batch, ys(minimum, sigma=sigma), 'b--')
ax2.scatter(batch, minimum, c='b', marker='s')

ax1.set_xscale('log', basex=2)
# ax1.set_yscale('log', basey=2)

ax1.set_xlabel('projection dimension of encoder')
ax1.set_ylabel('epochs',)
ax2.set_ylabel('log loss',)
ax1.grid()

ax1.vlines(x=168, ymin=-1, ymax=600, color='g', linestyle='dashed', label='original dimensionality')

ax1.legend(prop={'size': 10})

plot_location = 'plots/{}/{}_{}.png'.format('mimic', 'all', 'dim_comparison')
plt.savefig(plot_location, bbox_inches='tight', dpi=300)

