import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.argparser import pca_argparse
from common.utility import log_shapes, log_time, sep, \
    time_stp, load_processed_data, logger

from models.pca import PCABasic


def main(
        model,
        time_stamp,
        ally_classes,
        advr_1_classes,
        advr_2_classes,
        test_size,
        expl_var,
        expt,
        ):

    X, y_ally, y_advr_1, y_advr_2 = load_processed_data(
        expt, 'processed_data_X_y_ally_y_advr_y_advr_2.pkl')
    log_shapes(
        [X, y_ally, y_advr_1, y_advr_2],
        locals(),
        'Dataset loaded'
    )

    X_train, X_valid = train_test_split(
            X,
            test_size=test_size,
            stratify=pd.DataFrame(np.concatenate(
                (
                    y_ally.reshape(-1, ally_classes),
                    y_advr_1.reshape(-1, advr_1_classes),
                    y_advr_2.reshape(-1, advr_2_classes),
                ), axis=1)
            )
        )

    log_shapes(
        [
            X_train, X_valid,
        ],
        locals(),
        'Data size after train test split'
    )

    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_valid_normalized = scaler.transform(X_valid)

    log_shapes([X_train_normalized, X_valid_normalized], locals())

    pca = PCABasic(expl_var)
    X_train_pca = pca.train(X_train_normalized)
    X_valid_pca = pca.eval(X_valid_normalized)

    sep()
    logging.info('\nExplained Variance: {}\nNum Components: {}'.format(
        str(expl_var),
        pca.num_components,
    ))

    config_summary = 'dim_{}'.format(pca.num_components)

    model_ckpt = 'checkpoints/{}/{}_sklearn_model_{}_{}.pkl'.format(
            expt, model, time_stamp, config_summary)
    sep()
    logging.info('Saving: {}'.format(model_ckpt))
    joblib.dump(pca, model_ckpt)


if __name__ == "__main__":
    expt = 'titanic'
    model = 'pca_basic'
    marker = 'A'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    args = pca_argparse()
    main(
        model='pca_basic',
        time_stamp=fl_time,
        ally_classes=args['n_ally'],
        advr_1_classes=args['n_advr_1'],
        advr_2_classes=args['n_advr_2'],
        test_size=args['test_size'],
        expl_var=args['expl_var'],
        expt=args['expt'],
    )
    log_time('End', time_stp()[0])
    sep()
