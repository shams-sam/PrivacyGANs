import joblib
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.argparser import pca_argparse
from common.utility import log_time, sep, \
    time_stp, logger

from preprocessing import get_data

from models.pca import PCABasic


def main(
        model,
        time_stamp,
        ally_classes,
        advr_classes,
        test_size,
        expl_var,
        expt,
        ):

    X_normalized_train, X_normalized_valid,\
        y_ally_train, y_ally_valid, \
        y_advr_1_train, y_advr_1_valid, \
        y_advr_2_train, y_advr_2_valid = get_data(expt, test_size)

    pca = PCABasic(expl_var)
    X_train_pca = pca.train(X_normalized_train)
    X_valid_pca = pca.eval(X_normalized_valid)

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
    expt = 'mnist'
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
        advr_classes=args['n_advr'],
        test_size=args['test_size'],
        expl_var=args['expl_var'],
        expt=args['expt'],
    )
    log_time('End', time_stp()[0])
    sep()
