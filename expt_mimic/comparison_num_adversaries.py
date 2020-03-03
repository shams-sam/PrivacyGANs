import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

_12 = pkl.load(open('checkpoints/mimic/n_advr_ind_gan_training_history_02_12_2020_19_30_38.pkl', 'rb'))
_34 = pkl.load(open('checkpoints/mimic/n_advr_ind_gan_training_history_02_12_2020_20_01_01.pkl', 'rb'))
_56 = pkl.load(open('checkpoints/mimic/n_advr_ind_gan_training_history_02_12_2020_23_07_22.pkl', 'rb'))
_789 = pkl.load(open('checkpoints/mimic/n_advr_ind_gan_training_history_02_13_2020_13_54_53.pkl', 'rb'))
_n = pkl.load(open('checkpoints/mimic/n_advr_ind_gan_training_history_02_13_2020_18_31_15.pkl', 'rb'))

h = {1: _12, 2: _12, 3: _34, 4:_34, 5: _56, 6: _56, 7: _789, 8: _789, 9: _789, 10: _n}

orig = pkl.load(open('checkpoints/mimic/n_ind_training_history_02_02_2020_16_15_58.pkl', 'rb'))
loss_ind = orig['admission_type']['y_valid'][-1]


num_ally = []
loss = []
for idx, hist in h.items():
    num_ally.append(idx)
    loss.append(h[idx][idx-1]['advr_valid'][-1])


plt.figure(figsize=(5,3))
plt.plot(num_ally, loss, 'b--')
plt.scatter(num_ally, loss, color='b', marker='s', label='ally loss on EIGAN encoding')
plt.hlines(y=loss_ind, xmin=-1, xmax=11, color='r', linestyle='dashed')
plt.ylim(top=0.65, bottom=0.35)
plt.xlim(left=0)
plt.xlim(right=11)
plt.xlabel('number of adversaries')
plt.ylabel('log loss')
plt.legend(prop={'size':10})
plt.grid()
plt.text(2.7, 0.548, 'ally loss on original data', color='r', fontsize=12)


plot_location = 'plots/{}/{}_{}.png'.format('mimic', 'all', 'num_adversaries_comparison')
plt.savefig(plot_location, bbox_inches='tight', dpi=300)
