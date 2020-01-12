import numpy as np
from math import pi
import pickle as pkl


two_pi = 2*pi


def circle_points(r, n):
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, two_pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles


def main(
        expt,
        ):

    r = [10, 20]
    n = [32, 32]
    circles = circle_points(r, n)

    X, y_ally, y_advr = [], [], []
    ally_label = 0
    advr_label = 0

    for circle in circles:
        pos = circle[np.where(circle[:, 1]>= 0)]
        neg = circle[np.where(circle[:, 1]< 0)]
        X.append(pos)
        X.append(neg)
        y_ally.append([ally_label] * circle.shape[0])
        y_advr.append([advr_label] * pos.shape[0] + [advr_label+1] * neg.shape[0])

        ally_label += 1

    pkl.dump()


if __name__ == '__main__':
    expt = 'circle'
    main(
        expt=expt,
    )
