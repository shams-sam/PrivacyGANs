import common.config as cfg
from common.data import get_loader
from common.utility import log_shapes, load_processed_data, to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


def get_data(expt, dataset='mnist'):

    train_loader = get_loader(dataset, cfg.num_trains[dataset], True)
    valid_loader = get_loader(dataset, cfg.num_tests[dataset], False)

    for data, label in train_loader:
        X_train = data
        y_train = [label,
                   (label % 2 == 0),
                   (label >= 5)]

    for data, label in valid_loader:
        X_valid = data
        y_valid = [label,
                   (label % 2 == 0),
                   (label >= 5)]

    log_shapes(
        [X_train, X_valid] + [_ for _ in y_train] + [_ for _ in y_valid],
        locals(),
        'Dataset loaded'
    )

    return (
        X_train, X_valid,
        y_train, y_valid
    )
