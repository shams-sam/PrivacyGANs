from common.utility import sep
import argparse
import json
import logging

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


def autoencoder_argparse(debug=True):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--n-ally", type=int, nargs='+', required=True)
    ap.add_argument("--n-advr", type=int, nargs='+', required=False)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--test-size", type=float, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--n-epochs", type=int, required=True)
    ap.add_argument("--shuffle", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--expt", required=True)

    args = vars(ap.parse_args())

    if debug:
        sep()
        logging.info(json.dumps(args, indent=2))

    return args


def comparison_argparse(debug=True):
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", required=True)
    ap.add_argument("--n-ally", type=int, nargs='+', required=False)
    ap.add_argument("--n-advr", type=int, nargs='+', required=False)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--leaky", type=int, required=True)
    ap.add_argument("--epsilon", type=float, required=False)
    ap.add_argument("--test-size", type=float, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--n-epochs", type=int, required=True)
    ap.add_argument("--shuffle", type=int, required=True)
    ap.add_argument("--lr", type=float, required=False)
    ap.add_argument("--lr-ally", type=float, nargs='+', required=False)
    ap.add_argument("--lr-advr", type=float, nargs='+', required=False)
    ap.add_argument("--expt", required=True)
    ap.add_argument("--pca-ckpt", required=False)
    ap.add_argument("--autoencoder-ckpt", required=False)
    ap.add_argument("--encoder-ckpt", required=False)

    args = vars(ap.parse_args())

    if debug:
        sep()
        logging.info(json.dumps(args, indent=2))

    return args


def eigan_argparse(debug=True):
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", required=True)
    ap.add_argument("--n-gpu", type=int, required=True)
    ap.add_argument("--n-nodes", type=int, required=False)
    ap.add_argument("--n-ally", type=int, nargs='+', required=True)
    ap.add_argument("--n-advr", type=int, nargs='+', required=False,
    ap.add_argument("--n-channels", type=int, required=False)
    ap.add_argument("--n-filters", type=int, required=False)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--leaky", type=int, required=True)
    ap.add_argument("--activation", required=True)
    ap.add_argument("--test-size", type=float, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--n-epochs", type=int, required=True)
    ap.add_argument("--shuffle", type=int, required=True)
    ap.add_argument("--init-w", type=int, required=False)
    ap.add_argument("--lr-encd", type=float, required=True)
    ap.add_argument("--lr-ally", type=float, required=True)
    ap.add_argument("--lr-advr", type=float, required=False)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--g-reps", type=int, required=False)
    ap.add_argument("--d-reps", type=int, required=False)
    ap.add_argument("--num-allies", type=int, required=False)
    ap.add_argument("--num-adversaries", type=int, required=False)
    ap.add_argument("--expt", required=True)

    args=vars(ap.parse_args())

    if debug:
        sep()
        logging.info(json.dumps(args, indent=2))

    return args


def pca_argparse(debug=True):
    ap=argparse.ArgumentParser()

    ap.add_argument("--n-ally", type=int, required=True)
    ap.add_argument("--n-advr", type=int, nargs='+', required=False)
    ap.add_argument("--n-advr", type=int, nargs='+', required=False)
    ap.add_argument("--test-size", type=float, required=True)
    ap.add_argument("--expl-var", type=float, required=True)
    ap.add_argument("--expt", required=True)

    args=vars(ap.parse_args())

    if debug:
        sep()
        logging.info(json.dumps(args, indent=2))

    return args
