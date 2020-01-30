from datetime import datetime
import logging
import numpy as np
import pickle as pkl
import torch


# https://stackoverflow.com/questions/34980833/python-name-of-np-array-variable-as-string
def _namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def time_stp():
    now = datetime.now()

    return now.strftime("%m/%d/%Y %H:%M:%S"), now.strftime("%m_%d_%Y_%H_%M_%S")


def log_time(message, dt_string):
    logging.info('='*80)
    logging.info("+"*27 + "{}: {}".format(message, dt_string) + "+"*27)


def log_shapes(var_list, env, message=None):
    logging.info('='*80)
    if message:
        logging.info(message)
        logging.info('-'*80)
    for var in var_list:
        logging.info("{}: {}".format(_namestr(var, env), var.shape))


def sep(message=False):
    logging.info('='*80)
    if message:
        padding = int((80-len(message))/2)
        logging.info('{}{}{}'.format('+'*padding, message, '+'*padding))


def torch_device(device):
    dtype = torch.float
    if device == 'gpu':
        device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    logging.info('='*80)
    logging.info("DType: {}\nCuda available: {}\nDevice: {}".format(
        dtype.__str__(),
        torch.cuda.is_available().__str__(),
        device.__str__(),
    ))

    return device


def load_processed_data(expt, file='processed_data_X_y_ally_y_advr.pkl'):
    return pkl.load(
        open(
            'checkpoints/{}/{}'.format(expt, file),
            'rb'
        ),
    )


def logger(expt, model, time_stamp, marker):
    log_path = 'logs/{}/{}_{}_{}.log'.format(expt, model, time_stamp, marker)
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(message)s',
        level=logging.INFO
    )


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def laplaceMechanism(x, epsilon, device):
    x += torch.Tensor(np.random.laplace(0, 1.0/epsilon, x.shape)).to(device)
    return x
