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

h[4] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_21_58_14_F1_device_cuda_dim_4_hidden_8_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-0.9973_val_0.6641.pkl', 'rb'))
h[8] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_19_44_58_F1_device_cuda_dim_8_hidden_16_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.0133_val_0.6635.pkl', 'rb'))
h[16] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_19_48_28_F1_device_cuda_dim_16_hidden_32_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.0285_val_0.6626.pkl', 'rb'))
h[32] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_19_52_04_F1_device_cuda_dim_32_hidden_64_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.0477_val_0.6676.pkl', 'rb'))
h[64] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_19_55_38_F1_device_cuda_dim_64_hidden_128_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.1392_val_0.6711.pkl', 'rb'))
h[128] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_22_01_44_F1_device_cuda_dim_128_hidden_256_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.1871_val_0.6735.pkl', 'rb'))
h[256] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_22_05_21_F1_device_cuda_dim_256_hidden_512_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.2270_val_0.6833.pkl', 'rb'))
h[512] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_22_09_01_F1_device_cuda_dim_512_hidden_1024_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.2207_val_0.6884.pkl', 'rb'))
h[1024] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_22_12_24_F1_device_cuda_dim_1024_hidden_2048_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.1887_val_0.6728.pkl', 'rb'))
h[2048] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_22_16_06_F1_device_cuda_dim_2048_hidden_4086_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.1855_val_0.6702.pkl', 'rb'))
# h[4096] = pkl.load(open('checkpoints/titanic/eigan_training_history_02_13_2020_19_59_06_F1_device_cuda_dim_4096_hidden_8192_batch_1024_epochs_1001_lrencd_1e-05_lrally_1e-05_tr_-1.1288_val_0.6597.pkl', 'rb'))


sigma = 1

threshold = []
for batch, hist in h.items():
    threshold.append(0.99*np.mean(hist[2][-100:]))

converge = []
for batch, hist in h.items():
    for idx, _ in enumerate(hist[2]):
        if _ < threshold[int(np.log(batch)-2)]:
            converge.append(idx)
            break

minimum = []
for batch, hist in h.items():
    minimum.append(min(ys(hist[2], sigma)))

batch = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

fig = plt.figure(figsize=(5, 3))
ax1 = fig.add_subplot(111)

ax2 = ax1.twinx()
ax1.scatter(np.nan, np.nan, c='b', marker='s', label = 'final loss on convergence')
# ax1.plot(batch, ys(converge, sigma=sigma), 'r--')
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

plot_location = 'plots/{}/{}_{}.png'.format('titanic', 'all', 'dim_comparison')
plt.savefig(plot_location, bbox_inches='tight', dpi=300)
