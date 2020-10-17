import os
import joblib
import logging
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")  # noqa

from models.pca import PCABasic
from preprocessing import get_data
from common.utility import log_time, sep, \
    time_stp, logger
from common.argparser import pca_argparse
import common.config as cfg


def main(
        model,
        time_stamp,
        expl_var,
        expt,
):

    X_train, X_valid,\
        y_train, y_valid = get_data(expt)

    pca = PCABasic(expl_var)
    pca.train(X_train.reshape(cfg.num_trains[expt], -1))

    sep()
    logging.info('\nExplained Variance: {}\nNum Components: {}'.format(
        str(expl_var),
        pca.num_components,
    ))

    model_ckpt = 'ckpts/{}/models/{}_{}.pkl'.format(
        expt, model, marker)
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
        expl_var=args['expl_var'],
        expt=args['expt'],
    )
    log_time('End', time_stp()[0])
    sep()
