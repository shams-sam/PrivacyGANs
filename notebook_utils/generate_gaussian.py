import numpy as np


def generate_gaussian(VAR1, VAR2, MAX, BALANCE, MEAN=1):
    Bo = np.random.normal(loc=[MEAN, MEAN], scale=[VAR1, VAR2], size=[np.random.randint(8, MAX), 2])
    Ro = np.random.normal(loc=[MEAN, MEAN+1], scale=[VAR1, VAR2], size=[np.random.randint(8, MAX//BALANCE), 2])
    Bx = np.random.normal(loc=[MEAN+1, MEAN], scale=[VAR1, VAR2], size=[np.random.randint(8, MAX//BALANCE), 2])
    Rx = np.random.normal(loc=[MEAN+1, MEAN+1], scale=[VAR1, VAR2], size=[np.random.randint(8, MAX), 2])

    y_1 = np.hstack((
        np.zeros((Bo.shape[0])),
        np.ones((Ro.shape[0])),
        np.zeros((Bx.shape[0])),
        np.ones((Rx.shape[0]))
    )).astype(int).reshape(-1, 1)
    y_2 = np.hstack((
        np.zeros((Bo.shape[0])),
        np.zeros((Ro.shape[0])),
        np.ones((Bx.shape[0])),
        np.ones((Rx.shape[0]))
    )).astype(int).reshape(-1, 1)
    X = np.vstack((Bo, Ro, Bx, Rx))
    
    return X, y_1, y_2
