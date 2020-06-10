import matplotlib.pyplot as plt
import numpy as np
import torch


def class_plot(X, y_1, y_2, title=None, ax1=None):
    if not ax1:
        fig, (ax1) = plt.subplots(1, 1, figsize=(4, 4))

    markers = ['o', 'x', 'o', 'x']
    colors = ['b', 'r']

    for i in range(2):
        for j in range(2):
            tmp = X[np.intersect1d(
                np.where(y_1 == i)[0], np.where(y_2 == j)[0])]
            ax1.scatter(tmp[:, 0], tmp[:, 1],
                        c=colors[i], marker=markers[2*i+j])

    ax1.axis('equal')
    if title:
        ax1.set_title(title, y=-0.2)


def to_numpy(tensor):
    if type(tensor) == torch.Tensor:
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        tensor = tensor.detach().numpy()

    return tensor

