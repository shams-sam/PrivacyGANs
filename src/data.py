import os

import config as cfg
from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets, transforms


def get_subset_index(classes, split=0.3):
    _, _, idx_train, idx_valid = train_test_split(
        classes, list(range(len(classes))), test_size=split)

    return idx_valid


def get_loader(dataset, batch_size, train, img_size=64,
               subset=1, shuffle=True, data_folder=cfg.data_folder):
    kwargs = {}

    if dataset == 'mnist':
        def target_transform(target):
            a = 1-(target % 2)
            b = 1 if target >= 5 else 0
            return (target, a, b)
        dataset = datasets.MNIST(
            data_folder, train=train, download=cfg.download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]),
            target_transform=target_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'cifar_100':
        from cifar_100 import CIFAR100

        def target_transform(target):
            coarse = CIFAR100.get_coarse_class_ids([target])
            return (target, coarse[0])
        dataset = datasets.CIFAR100(
            data_folder, train=train, download=cfg.download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2761))
            ]),
            target_transform=target_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'celeba':
        if train:
            train = 'train'
        else:
            train = 'valid'

        def target_transform(target):
            return [target[i] for i in range(len(target))]

        dataset = datasets.CelebA(
            data_folder, split=train, download=cfg.download,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                transforms.Normalize((0.5063, 0.4258, 0.3832),
                                     (0.3106, 0.2904, 0.2897))]),
            target_transform=target_transform)
        if subset < 1:
            indices = get_subset_index(dataset.attr, subset)
            dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'facescrub':
        from facescrub import FaceScrub

        def target_transform(target):
            gender = FaceScrub.get_gender([target])
            return (target, gender[0])
        subfolder = 'FaceScrub/split/{}'.format('train' if train else 'val')
        dataset = datasets.ImageFolder(
            os.path.join(data_folder, subfolder),
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.6106, 0.4647, 0.3955),
                                     (0.2673, 0.2564, 0.2761))
            ]),
            target_transform=target_transform,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return dataloader
