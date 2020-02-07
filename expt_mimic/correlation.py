import pickle as pkl
import numpy as np 
import pandas as pd 
import seaborn as sns

X, targets = pkl.load(open('checkpoints/mimic/processed_data_X_targets.pkl', 'rb'))
corr = np.zeros((11, 11))

labels = [
    'admission_type',
    'hospital_expire_flag',
    'expire_flag',
    'gender',
    'language',
    'marital_status',
    'religion',
    'insurance',
    'discharge_location',
    'admission_location',
    'ethnicity',
]

r = 0
for l1 in labels:
    c = 0
    for l2 in labels:
        corr[r, c] = np.corrcoef(targets[l1], targets[l2])[0, 1] 
        c+=1
    r += 1

df = pd.DataFrame(corr, index=labels, columns=labels)

plt = sns.heatmap(df, annot=True, fmt=".1f")
plt.figure.savefig('plots/mimic/all_correlation.png', dpi=300, bbox_inches='tight')