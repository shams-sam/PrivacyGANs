import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib

matplotlib.rcParams.update({'font.size': 11})


s = pkl.load(open('checkpoints/mimic/n_eigan_training_history_02_03_2020_00_59_27_B_device_cuda_dim_256_hidden_512_batch_16384_epochs_1001_ally_0_encd_0.0276_advr_0.5939.pkl','rb'))
u = pkl.load(open('checkpoints/mnist/eigan_training_history_02_05_2020_19_54_36_A_device_cuda_dim_1024_hidden_2048_batch_4096_epochs_501_lrencd_0.01_lrally_1e-05_lradvr_1e-05_tr_0.4023_val_1.7302.pkl', 'rb'))

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(121)
ax1.plot(s[0], s[2], 'r', label='encoder loss')
ax1.set_title('(a)', y=-0.25)
ax1.plot(np.nan, 'b', label = 'adversary loss')
ax1.legend()
ax1.set_xlabel('epochs')
ax1.set_ylabel('encoder loss')
ax1.grid()
ax1.set_xlim(right=500)
ax2 = ax1.twinx()
ax2.plot(s[0], s[6], 'b')
ax2.set_ylabel('adversary loss')

ax3 = fig.add_subplot(122)
ax3.plot(u[0], u[2], 'r', label='encoder loss')
ax3.plot(np.nan, 'b', label = 'adversary loss')
ax4 = ax3.twinx()
ax4.plot(u[0], u[6], 'b')
ax3.set_title('(b)', y=-0.25)
ax3.legend()
ax3.set_xlabel('epochs')
ax3.set_ylabel('encoder loss')
ax3.grid()
ax4.set_ylabel('adversary loss')

fig.subplots_adjust(wspace=0.4)

plot_location = 'plots/{}/{}_{}.png'.format('mimic', 'all', 'training_comparison')
plt.savefig(plot_location, bbox_inches='tight')
