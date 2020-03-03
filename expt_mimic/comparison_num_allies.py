import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

_13 = pkl.load(open('checkpoints/mimic/n_ind_gan_training_history_02_03_2020_17_41_09.pkl', 'rb'))
_28 = pkl.load(open('checkpoints/mimic/n_ind_gan_training_history_02_05_2020_00_23_29.pkl', 'rb'))
_4 = pkl.load(open('checkpoints/mimic/n_ind_gan_training_history_02_03_2020_20_09_39.pkl', 'rb'))
_5 = pkl.load(open('checkpoints/mimic/n_ind_gan_training_history_02_04_2020_00_21_50.pkl', 'rb'))
_67 = pkl.load(open('checkpoints/mimic/n_ind_gan_training_history_02_04_2020_05_30_09.pkl', 'rb'))
_8 = pkl.load(open('checkpoints/mimic/n_ind_gan_training_history_02_05_2020_00_23_29.pkl', 'rb'))
_9 = pkl.load(open('checkpoints/mimic/n_ind_gan_training_history_02_05_2020_18_09_03.pkl', 'rb'))
_n = pkl.load(open('checkpoints/mimic/n_ind_gan_training_history_02_04_2020_20_13_29.pkl', 'rb'))


h = {1: _13, 2: _28, 3: _13, 4:_4, 5: _5, 6: _67, 7: _67, 8: _8, 9: _9, 10: _n}

orig = pkl.load(open('checkpoints/mimic/n_ind_training_history_02_02_2020_16_15_58.pkl', 'rb'))
loss_ind = orig['admission_type']['y_valid'][-1]


num_ally = []
loss = []
for idx, hist in h.items():
    num_ally.append(idx)
    loss.append(h[idx][idx-1]['advr_valid'][-1])


plt.figure(figsize=(5,3))
plt.plot(num_ally, loss, 'b--')
plt.scatter(num_ally, loss, color='b', marker='s', label='adversary loss on EIGAN encoding')
plt.hlines(y=loss_ind, xmin=-1, xmax=11, color='r', linestyle='dashed')
plt.text(2, 0.542, 'adversary loss on original data', fontsize=12, color='r')
plt.ylim(top=0.65)
plt.ylim(bottom=0.50)
plt.xlim(left=0)
plt.xlim(right=11)
plt.xlabel('number of allies')
plt.ylabel('log loss')
plt.legend(prop={'size':10})
plt.grid()


plot_location = 'plots/{}/{}_{}.png'.format('mimic', 'all', 'num_allies_comparison')
plt.savefig(plot_location, bbox_inches='tight', dpi=300)
