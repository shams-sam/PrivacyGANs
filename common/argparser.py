import argparse
import json
import logging

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.utility import sep


def autoencoder_argparse(debug=True):
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", required=True,
                    help="gpu/cpu")
    ap.add_argument("--n-ally", type=int, required=True,
                    help="number of ally classes")
    ap.add_argument("--n-advr", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--n-advr-1", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--n-advr-2", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--dim", type=int, required=True,
                    help="encoding dimension")
    ap.add_argument("--test-size", type=float, required=True,
                    help="size of test set")
    ap.add_argument("--batch-size", type=int, required=True,
                    help="batch size")
    ap.add_argument("--n-epochs", type=int, required=True,
                    help="number of epochs")
    ap.add_argument("--shuffle", type=int, required=True,
                    help="to shuffle or not")
    ap.add_argument("--lr", type=float, required=True,
                    help="learning rate")
    ap.add_argument("--expt", required=True,
                    help="experiment/dataset name")

    args = vars(ap.parse_args())

    if debug:
        sep()
        logging.info(json.dumps(args, indent=2))

    return args


def comparison_argparse(debug=True):
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", required=True,
                    help="gpu/cpu")
    ap.add_argument("--n-ally", type=int, required=False,
                    help="number of ally classes")
    ap.add_argument("--n-advr", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--n-advr-1", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--n-advr-2", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--dim", type=int, required=True,
                    help="encoding dimension")
    ap.add_argument("--hidden-dim", type=int, required=True,
                    help="hidden layer dimension")
    ap.add_argument("--leaky", type=int, required=True,
                    help="leaky relu or not")
    ap.add_argument("--epsilon", type=float, required=False,
                    help="epsilon value")
    ap.add_argument("--test-size", type=float, required=True,
                    help="size of test set")
    ap.add_argument("--batch-size", type=int, required=True,
                    help="batch size")
    ap.add_argument("--n-epochs", type=int, required=True,
                    help="number of epochs")
    ap.add_argument("--shuffle", type=int, required=True,
                    help="to shuffle or not")
    ap.add_argument("--lr", type=float, required=False,
                    help="learning rate")
    ap.add_argument("--lr-ally", type=float, required=False,
                    help="learning rate for ally")
    ap.add_argument("--lr-advr", type=float, required=False,
                    help="learning rate for advr")
    ap.add_argument("--lr-advr-1", type=float, required=False,
                    help="learning rate for advr")
    ap.add_argument("--lr-advr-2", type=float, required=False,
                    help="learning rate for advr")
    ap.add_argument("--expt", required=True,
                    help="experiment/dataset name")
    ap.add_argument("--pca-ckpt", required=False,
                    help="pca checkpoint")
    ap.add_argument("--autoencoder-ckpt", required=False,
                    help="autoencoder checkpoint")
    ap.add_argument("--encoder-ckpt", required=False,
                    help="encoder checkpoint")

    args = vars(ap.parse_args())

    if debug:
        sep()
        logging.info(json.dumps(args, indent=2))

    return args


def eigan_argparse(debug=True):
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", required=True,
                    help="gpu/cpu")
    ap.add_argument("--n-gpu", type=int, required=True,
                    help="number of gpus")
    ap.add_argument("--n-ally", type=int, required=True,
                    help="number of ally classes")
    ap.add_argument("--n-advr", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--n-advr-1", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--n-advr-2", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--n-channels", type=int, required=False,
                    help="number of input channels")
    ap.add_argument("--n-filters", type=int, required=False,
                    help="number of filter/output channels")
    ap.add_argument("--dim", type=int, required=True,
                    help="encoding dimension")
    ap.add_argument("--hidden-dim", type=int, required=True,
                    help="hidden layer dimension")
    ap.add_argument("--leaky", type=int, required=True,
                    help="leaky relu or not")
    ap.add_argument("--activation", required=True,
                    help="final activation sigmoid/tanh")
    ap.add_argument("--test-size", type=float, required=True,
                    help="size of test set")
    ap.add_argument("--batch-size", type=int, required=True,
                    help="batch size")
    ap.add_argument("--n-epochs", type=int, required=True,
                    help="number of epochs")
    ap.add_argument("--shuffle", type=int, required=True,
                    help="to shuffle or not")
    ap.add_argument("--init-w", type=int, required=False,
                    help="to init weight or not")
    ap.add_argument("--lr-encd", type=float, required=True,
                    help="learning rate for encoder")
    ap.add_argument("--lr-ally", type=float, required=True,
                    help="learning rate for ally")
    ap.add_argument("--lr-advr", type=float, required=False,
                    help="learning rate for advr")
    ap.add_argument("--lr-advr-1", type=float, required=False,
                    help="learning rate for advr")
    ap.add_argument("--lr-advr-2", type=float, required=False,
                    help="learning rate for advr")
    ap.add_argument("--alpha", type=float, required=True,
                    help="trade off between privacy and prediction")
    ap.add_argument("--g-reps", type=int, required=False,
                    help="generator reps")
    ap.add_argument("--d-reps", type=int, required=False,
                    help="discriminator reps")
    ap.add_argument("--num-allies", type=int, required=False,
                    help="number of allies")
    ap.add_argument("--num-adversaries", type=int, required=False,
                    help="number of adversaries")
    ap.add_argument("--expt", required=True,
                    help="experiment/dataset name")

    args = vars(ap.parse_args())

    if debug:
        sep()
        logging.info(json.dumps(args, indent=2))

    return args


def pca_argparse(debug=True):
    ap = argparse.ArgumentParser()

    ap.add_argument("--n-ally", type=int, required=True,
                    help="number of ally classes")
    ap.add_argument("--n-advr", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--n-advr-1", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--n-advr-2", type=int, required=False,
                    help="number of advr classes")
    ap.add_argument("--test-size", type=float, required=True,
                    help="size of test set")
    ap.add_argument("--expl-var", type=float, required=True,
                    help="explained variance")
    ap.add_argument("--expt", required=True,
                    help="experiment/dataset name")

    args = vars(ap.parse_args())

    if debug:
        sep()
        logging.info(json.dumps(args, indent=2))

    return args
