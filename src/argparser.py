import os
import sys

import argparse
import json
import logging
from utils import booltype


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse(debug=True):
    ap = argparse.ArgumentParser()
    ap.add_argument("--expt", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--gpu-id", type=int, nargs='+', required=True)
    ap.add_argument("--train-nets", type=str, nargs='+', required=False)
    ap.add_argument("--optimizer", required=False)
    ap.add_argument("--arch", type=str, required=False)
    ap.add_argument("--resize", type=int, required=False)
    ap.add_argument("--num-layers", type=int, required=False)
    ap.add_argument("--n-nodes", type=int, required=False)
    ap.add_argument("--n-classes", type=int, nargs='+', required=False)
    ap.add_argument("--n-ally", type=int, nargs='+', required=False)
    ap.add_argument("--n-advr", type=int, nargs='+', required=False)
    ap.add_argument("--n-channels", type=int, required=False)
    ap.add_argument("--n-filters", type=int, required=False)
    ap.add_argument("--dim", type=int, required=False)
    ap.add_argument("--hidden-dim", type=int, required=False)
    ap.add_argument("--leaky", type=int, required=False)
    ap.add_argument("--activation", required=False)
    ap.add_argument("--expl-var", type=float, required=False)
    ap.add_argument("--img-size", type=int, required=False)
    ap.add_argument("--batch-size", type=int, required=False)
    ap.add_argument("--test-batch-size", type=int, required=False)
    ap.add_argument("--subset", type=float, required=False)
    ap.add_argument("--n-epochs", type=int, required=False)
    ap.add_argument("--shuffle", type=int, required=False)
    ap.add_argument("--init-w", type=booltype, required=False)
    ap.add_argument("--load-w", type=booltype, required=False)
    ap.add_argument("--ckpt-g", type=str, required=False)
    ap.add_argument("--ckpt-d", type=str, required=False)
    ap.add_argument("--ckpt-clfs", type=str, nargs='+', required=False)
    ap.add_argument("--lr", type=float, required=False)
    ap.add_argument("--lr-g", type=float, required=False)
    ap.add_argument("--lr-d", type=float, required=False)
    ap.add_argument("--lr-clfs", type=float, nargs='+', required=False)
    ap.add_argument("--ei-array", type=float, nargs='+', required=False)
    ap.add_argument("--weight-decays", type=float, nargs='+', required=False)
    ap.add_argument("--gamma", type=float, required=False)
    ap.add_argument("--milestones", type=int, nargs='+', required=False)
    ap.add_argument("--alpha", type=float, required=False)
    ap.add_argument("--save-ckpts", type=int, nargs="+", required=False)
    ap.add_argument("--marker", required=False)

    args = vars(ap.parse_args())

    return Struct(**args)
