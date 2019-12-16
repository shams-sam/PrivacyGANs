from datetime import datetime
import pickle as pkl
import torch


# https://stackoverflow.com/questions/34980833/python-name-of-np-array-variable-as-string
def _namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def time_stp(print=False):
    now = datetime.now()
    if print:
        return now.strftime("%m/%d/%Y %H:%M:%S")
    return now.strftime("%m_%d_%Y_%H_%M_%S")


def print_time(message):
    dt_string = time_stp(True)

    print('='*80)
    print("+"*28 + "{}: {}".format(message, dt_string) + "+"*28)


def print_shapes(var_list, env, message=None):
    print('='*80)
    if message:
        print(message)
        print('-'*80)
    for var in var_list:
        print("{}: {}".format(_namestr(var, env), var.shape))


def torch_device(device):
    dtype = torch.float
    if device == 'gpu':
        device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    print('='*80)
    print("DType: {}\nCuda available: {}\nDevice: {}".format(
        dtype.__str__(),
        torch.cuda.is_available().__str__(),
        device.__str__(),
    ))

    return device

def load_processed_data(expt):
    return pkl.load(
        open(
            'checkpoints/{}/processed_data_X_y_ally_y_advr.pkl'.format(expt),
            'rb'
        ),
    )
