import torch
import numpy as np


def federated(encoders, weights, global_params, phi, device):
    num_encoders = len(encoders)
    params = [_.named_parameters() for _ in encoders]
    dict_params = [dict(param) for param in params]

    if not len(global_params):
        for _, param in dict_params[0].items():
            global_params[_] = torch.zeros(param.size()).to(device)

    for key, param in global_params.items():
        size = param.size()
        param = param.flatten()
        len_param = param.size()[0]
        idx = torch.randperm(len_param)[:int(phi*len_param)]
        weight = torch.zeros(param.size()).to(device)
        for node_idx in range(num_encoders):
            weight[idx] += weights[node_idx] \
                           * dict_params[node_idx][key].data.flatten()[idx]
        param = weight
        param = param.reshape(size)
        global_params[key] = param
        for _ in dict_params:
            tmp = _[key].data
            size = tmp.size()
            tmp = tmp.flatten()
            len_param = tmp.size()[0]
            idx = torch.randperm(len_param)[:int(phi*len_param)]
            tmp[idx] = global_params[key].flatten()[idx]
            tmp = tmp.reshape(size)
            _[key].data.copy_(tmp) 

    return encoders
