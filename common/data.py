import common.config as cfg
import torch
from torchvision import datasets, transforms


def get_loader(dataset, batch_size, train, shuffle=True):
    kwargs = {}
    if dataset == 'mnist':
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=train,
                           download=cfg.download,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'cifar_10':
        return torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=train,
                             download=cfg.download,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'fmnist':
        return torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=train,
                                  download=cfg.download,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.2861,),
                                                           (0.3530,))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
