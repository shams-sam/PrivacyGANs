from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .eigan import Encoder, Discriminator
from .federated import federated
from .utility import class_plot


def centralized(
        X_train,
        X_valid,
        y_1_train,
        y_1_valid,
        y_2_train,
        y_2_valid,
        input_size,
        hidden_size,
        output_size,
        alpha,
        lr_encd,
        lr_1,
        lr_2,
        w_1,
        w_2,
        train_loader,
        n_iter_gan,
        device,
        plot=True,
        debug=True,
        eps=-np.log(0.5)
):
    encoder = Encoder(input_size=input_size,
                      hidden_size=hidden_size,
                      output_size=input_size).to(device)
    clf_1 = Discriminator(input_size=input_size,
                          hidden_size=hidden_size,
                          output_size=output_size).to(device)
    clf_2 = Discriminator(input_size=input_size,
                          hidden_size=hidden_size,
                          output_size=output_size).to(device)

    bce_1_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor(w_1)).to(device)
    bce_2_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor(w_2)).to(device)

    optimizer = torch.optim.Adam
    encd_optimizer = optimizer(encoder.parameters(), lr=lr_encd)
    clf_1_optimizer = optimizer(clf_1.parameters(), lr=lr_1)
    clf_2_optimizer = optimizer(clf_2.parameters(), lr=lr_2)

    if debug:
        print(
            "epoch \t encoder_train \t encoder_valid \t 1_train \t 1_valid "
            "\t 2_train \t 2_valid"
        )

    g_epoch = []
    enc_train = []
    enc_valid = []
    clf_1_train = []
    clf_1_valid = []
    clf_2_train = []
    clf_2_valid = []

    encoder.train()
    clf_1.train()
    clf_2.train()

    for epoch in tqdm(range(n_iter_gan)):
        for _, batch in enumerate(train_loader):
            x = batch[0]
            y1 = batch[1]
            y2 = batch[2]

            x_ = encoder(x)
            y1_ = clf_1(x_)
            y2_ = clf_2(x_)

            clf_1_train_loss = bce_1_loss(y1_, y1)
            clf_2_train_loss = bce_2_loss(y2_, y2)
            encd_train_loss = clf_1_train_loss \
                - alpha * (clf_2_train_loss - eps)

            encd_optimizer.zero_grad()
            encd_train_loss.backward()
            encd_optimizer.step()

            x_ = encoder(x)
            y1_ = clf_1(x_)
            clf_1_train_loss = bce_1_loss(y1_, y1)

            clf_1_optimizer.zero_grad()
            clf_1_train_loss.backward()
            clf_1_optimizer.step()

            x_ = encoder(x)
            y2_ = clf_2(x_)
            clf_2_train_loss = bce_2_loss(y2_, y2)

            clf_2_optimizer.zero_grad()
            clf_2_train_loss.backward()
            clf_2_optimizer.step()

        X_valid_ = encoder(X_valid)
        y_1_valid_ = clf_1(X_valid_)
        y_2_valid_ = clf_2(X_valid_)

        clf_1_valid_loss = bce_1_loss(y_1_valid_, y_1_valid)
        clf_2_valid_loss = bce_2_loss(y_2_valid_, y_2_valid)
        encd_valid_loss = clf_1_valid_loss - alpha * (clf_2_valid_loss - eps)

        if plot:
            g_epoch.append(epoch)
            enc_train.append(encd_train_loss.item())
            enc_valid.append(encd_valid_loss.item())
            clf_1_train.append(clf_1_train_loss.item())
            clf_1_valid.append(clf_1_valid_loss.item())
            clf_2_train.append(clf_2_train_loss.item())
            clf_2_valid.append(clf_2_valid_loss.item())

        if epoch % 500 != 0 and (debug or plot):
            continue

        if debug:
            print('{} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f}'.format(
                epoch, 
                encd_train_loss.item(),
                encd_valid_loss.item(),
                clf_1_train_loss.item(),
                clf_1_valid_loss.item(),
                clf_2_train_loss.item(),
                clf_2_valid_loss.item()
            ))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    X_train_ = encoder(X_train)
    class_plot(X_train_.cpu().detach(),
               np.argmax(y_1_train.cpu(), axis=1),
               np.argmax(y_2_train.cpu(), axis=1),
               'train @ {}'.format(epoch), ax1)
    X_valid_ = encoder(X_valid)
    class_plot(X_valid_.cpu().detach(),
               np.argmax(y_1_valid.cpu(), axis=1),
               np.argmax(y_2_valid.cpu(), axis=1),
               'valid @ {}'.format(epoch), ax2)
    plt.show()

    if plot:
        plt.plot(g_epoch, enc_train, 'r', g_epoch, enc_valid, 'r--')
        plt.plot(g_epoch, clf_1_train, 'b', g_epoch, clf_1_valid, 'b--')
        plt.plot(g_epoch, clf_2_train, 'g', g_epoch, clf_2_valid, 'g--')
        plt.legend([
            'encoder train', 'encoder valid',
            'clf_1 train', 'clf_1 valid',
            'clf_2 train', 'clf_2 valid',
        ])
        plt.title("EIGAN @ {}".format(epoch))
        plt.show()

        return encoder


def centralized_3(
        X_train,
        X_valid,
        y_1_train,
        y_1_valid,
        y_2_train,
        y_2_valid,
        y_3_train,
        y_3_valid,
        input_size,
        hidden_size,
        output_size,
        alpha,
        lr_encd,
        lr_1,
        lr_2,
        lr_3,
        w_1,
        w_2,
        w_3,
        train_loader,
        n_iter_gan,
        device,
        plot=True,
        debug=True,
        eps2=-np.log(0.5),
        eps3=-np.log(0.5),
):
    encoder = Encoder(input_size=input_size,
                      hidden_size=hidden_size,
                      output_size=input_size).to(device)
    clf_1 = Discriminator(input_size=input_size,
                          hidden_size=hidden_size,
                          output_size=output_size[0]).to(device)
    clf_2 = Discriminator(input_size=input_size,
                          hidden_size=hidden_size,
                          output_size=output_size[1]).to(device)
    clf_3 = Discriminator(input_size=input_size,
                          hidden_size=hidden_size,
                          output_size=output_size[2]).to(device)

    bce_1_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor(w_1)).to(device)
    bce_2_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor(w_2)).to(device)
    bce_3_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor(w_3)).to(device)

    optimizer = torch.optim.Adam
    encd_optimizer = optimizer(encoder.parameters(), lr=lr_encd)
    clf_1_optimizer = optimizer(clf_1.parameters(), lr=lr_1)
    clf_2_optimizer = optimizer(clf_2.parameters(), lr=lr_2)
    clf_3_optimizer = optimizer(clf_3.parameters(), lr=lr_3)

    if debug:
        print(
            "epoch \t encoder_train \t encoder_valid \t 1_train \t 1_valid "
            "\t 2_train \t 2_valid \t 3_train \t 3_valid"
        )

    g_epoch = []
    enc_train = []
    enc_valid = []
    clf_1_train = []
    clf_1_valid = []
    clf_2_train = []
    clf_2_valid = []
    clf_3_train = []
    clf_3_valid = []
    
    encoder.train()
    clf_1.train()
    clf_2.train()
    clf_3.train()

    for epoch in tqdm(range(n_iter_gan)):
        for _, batch in enumerate(train_loader):
            x = batch[0]
            y1 = batch[1]
            y2 = batch[2]
            y3 = batch[3]

            x_ = encoder(x)
            y1_ = clf_1(x_)
            y2_ = clf_2(x_)
            y3_ = clf_3(x_)

            clf_1_train_loss = bce_1_loss(y1_, y1)
            clf_2_train_loss = bce_2_loss(y2_, y2)
            clf_3_train_loss = bce_3_loss(y3_, y3)
            encd_train_loss = clf_1_train_loss \
                - alpha * (clf_2_train_loss - eps2) \
                - alpha * (clf_3_train_loss - eps3)

            encd_optimizer.zero_grad()
            encd_train_loss.backward()
            encd_optimizer.step()

            x_ = encoder(x)
            y1_ = clf_1(x_)
            clf_1_train_loss = bce_1_loss(y1_, y1)

            clf_1_optimizer.zero_grad()
            clf_1_train_loss.backward()
            clf_1_optimizer.step()

            x_ = encoder(x)
            y2_ = clf_2(x_)
            clf_2_train_loss = bce_2_loss(y2_, y2)

            clf_2_optimizer.zero_grad()
            clf_2_train_loss.backward()
            clf_2_optimizer.step()

            x_ = encoder(x)
            y3_ = clf_3(x_)
            clf_3_train_loss = bce_3_loss(y3_, y3)

            clf_3_optimizer.zero_grad()
            clf_3_train_loss.backward()
            clf_3_optimizer.step()

        X_valid_ = encoder(X_valid)
        y_1_valid_ = clf_1(X_valid_)
        y_2_valid_ = clf_2(X_valid_)
        y_3_valid_ = clf_3(X_valid_)

        clf_1_valid_loss = bce_1_loss(y_1_valid_, y_1_valid)
        clf_2_valid_loss = bce_2_loss(y_2_valid_, y_2_valid)
        clf_3_valid_loss = bce_3_loss(y_3_valid_, y_3_valid)
        encd_valid_loss = clf_1_valid_loss \
            - alpha * (clf_2_valid_loss - eps2) \
            - alpha * (clf_3_valid_loss - eps3)

        if plot:
            g_epoch.append(epoch)
            enc_train.append(encd_train_loss.item())
            enc_valid.append(encd_valid_loss.item())
            clf_1_train.append(clf_1_train_loss.item())
            clf_1_valid.append(clf_1_valid_loss.item())
            clf_2_train.append(clf_2_train_loss.item())
            clf_2_valid.append(clf_2_valid_loss.item())
            clf_3_train.append(clf_3_train_loss.item())
            clf_3_valid.append(clf_3_valid_loss.item())

        if epoch % 20 != 0 and (debug or plot):
            continue

        if debug:
            print('{} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} '
                  '\t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f}'.format(
                epoch, 
                encd_train_loss.item(),
                encd_valid_loss.item(),
                clf_1_train_loss.item(),
                clf_1_valid_loss.item(),
                clf_2_train_loss.item(),
                clf_2_valid_loss.item(),
                clf_3_train_loss.item(),
                clf_3_valid_loss.item()
            ))

    if plot:
        plt.plot(g_epoch, enc_train, 'r', g_epoch, enc_valid, 'r--')
        plt.plot(g_epoch, clf_1_train, 'b', g_epoch, clf_1_valid, 'b--')
        plt.plot(g_epoch, clf_2_train, 'g', g_epoch, clf_2_valid, 'g--')
        plt.plot(g_epoch, clf_3_train, 'y', g_epoch, clf_3_valid, 'y--')
        plt.legend([
            'encoder train', 'encoder valid',
            'clf_1 train', 'clf_1 valid',
            'clf_2 train', 'clf_2 valid',
            'clf_3 train', 'clf_3 valid',
        ])
        plt.title("EIGAN @ {}".format(epoch))
        plt.show()

        return encoder


def distributed(
        num_nodes,
        phi,
        delta,
        X_trains,
        X_valids,
        y_1_trains,
        y_1_valids,
        y_2_trains,
        y_2_valids,
        input_size,
        hidden_size,
        output_size,
        alpha,
        lr_encd,
        lr_1,
        lr_2,
        w_1,
        w_2,
        train_loaders,
        n_iter_gan,
        device,
        global_params,
        plot=True,
        debug=True,
        eps=-np.log(0.5),
):
    encoders = []
    clfs_1 = []
    clfs_2 = []
    bce_1_losses = []
    bce_2_losses = []
    optimizer = torch.optim.Adam
    encd_optimizers = []
    clf_1_optimizers = []
    clf_2_optimizers = []

    for _ in range(num_nodes):
        encoders.append(Encoder(input_size=input_size, hidden_size=hidden_size,
                                output_size=input_size).to(device))
        clfs_1.append(Discriminator(input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=output_size).to(device))
        clfs_2.append(Discriminator(input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=output_size).to(device))

        bce_1_losses.append(torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(w_1[_])).to(device))
        bce_2_losses.append(torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(w_2[_])).to(device))

        encd_optimizers.append(optimizer(encoders[_].parameters(), lr=lr_encd))
        clf_1_optimizers.append(optimizer(clfs_1[_].parameters(), lr=lr_1))
        clf_2_optimizers.append(optimizer(clfs_2[_].parameters(), lr=lr_2))

    if debug:
        print("epoch \t node \t encoder_train \t encoder_valid \t 1_train "
              "\t 1_valid \t 2_train \t 2_valid"
        )

    g_epoch = defaultdict(list)
    enc_train = defaultdict(list)
    enc_valid = defaultdict(list)
    clf_1_train = defaultdict(list)
    clf_1_valid = defaultdict(list)
    clf_2_train = defaultdict(list)
    clf_2_valid = defaultdict(list)

    for _ in encoders:
        _.train()
    for _ in clfs_1:
        _.train()
    for _ in clfs_2:
        _.train()

    d = 0

    weights = np.zeros((num_nodes))
    for idx in range(num_nodes):
        weights[idx] = len(X_trains[idx])
    weights = weights/weights.sum()
    encoders = federated(encoders, weights, global_params, 1, device)
    
    for epoch in range(n_iter_gan):
        if d == delta and delta != 0 and phi != 0:
            print('Aggregating on epoch: {}...'.format(epoch))
            encoders = federated(encoders, weights, global_params,
                                 phi, device)
            d = 0
        d += 1
        for node_idx in range(num_nodes):
            # others = [idx for idx in range(num_nodes) if idx != node_idx]
            for _, batch in enumerate(train_loaders[node_idx]):
                x = batch[0].to(device)
                y1 = batch[1].to(device)
                y2 = batch[2].to(device)

                x_ = encoders[node_idx](x)
                y1_ = clfs_1[node_idx](x_)
                y2_ = clfs_2[node_idx](x_)

                clf_1_train_loss = bce_1_losses[node_idx](y1_, y1)
                clf_2_train_loss = bce_2_losses[node_idx](y2_, y2)
                encd_train_loss = clf_1_train_loss \
                    - alpha * (clf_2_train_loss - eps)

                encd_optimizers[node_idx].zero_grad()
                encd_train_loss.backward()
                encd_optimizers[node_idx].step()

                x_ = encoders[node_idx](x)
                y1_ = clfs_1[node_idx](x_)
                clf_1_train_loss = bce_1_losses[node_idx](y1_, y1)

                clf_1_optimizers[node_idx].zero_grad()
                clf_1_train_loss.backward()
                clf_1_optimizers[node_idx].step()

                x_ = encoders[node_idx](x)
                y2_ = clfs_2[node_idx](x_)
                clf_2_train_loss = bce_2_losses[node_idx](y2_, y2)

                clf_2_optimizers[node_idx].zero_grad()
                clf_2_train_loss.backward()
                clf_2_optimizers[node_idx].step()
            
            X_valid_ = encoders[node_idx](X_valids[node_idx].to(device))
            y_1_valid_ = clfs_1[node_idx](X_valid_) 
            y_2_valid_ = clfs_2[node_idx](X_valid_)

            clf_1_valid_loss = bce_1_losses[node_idx](
                y_1_valid_, y_1_valids[node_idx].to(device))
            clf_2_valid_loss = bce_2_losses[node_idx](
                y_2_valid_, y_2_valids[node_idx].to(device))
            
            encd_valid_loss = clf_1_valid_loss \
                - alpha * (clf_2_valid_loss - eps)
            
            if plot:
                g_epoch[node_idx].append(epoch)
                enc_train[node_idx].append(encd_train_loss.item())
                enc_valid[node_idx].append(encd_valid_loss.item())
                clf_1_train[node_idx].append(clf_1_train_loss.item())
                clf_1_valid[node_idx].append(clf_1_valid_loss.item())
                clf_2_train[node_idx].append(clf_2_train_loss.item())
                clf_2_valid[node_idx].append(clf_2_valid_loss.item())

            if epoch % 20 != 0:
                continue

            if debug:
                print('{} \t {} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} '
                      '\t {:.8f} \t {:.8f}'.format(
                          epoch, 
                          node_idx,
                          encd_train_loss.item(),
                          encd_valid_loss.item(),
                          clf_1_train_loss.item(),
                          clf_1_valid_loss.item(),
                          clf_2_train_loss.item(),
                          clf_2_valid_loss.item()
                      ))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
            X_train_ = encoders[node_idx](X_trains[node_idx].to(device))
            class_plot(X_train_.detach().cpu().numpy(),
                       np.argmax(y_1_trains[node_idx], axis=1),
                       np.argmax(y_2_trains[node_idx], axis=1),
                       'node:{}, train @ {}'.format(node_idx, epoch), ax1)
            X_valid_ = encoders[node_idx](X_valids[node_idx].to(device))
            class_plot(X_valid_.detach().cpu().numpy(),
                       np.argmax(y_1_valids[node_idx], axis=1), 
                       np.argmax(y_2_valids[node_idx], axis=1), 
                       'node:{}, valid @ {}'.format(node_idx, epoch), ax2)
            plt.show()

    if plot:
        for node_idx in range(num_nodes):
            plt.plot(g_epoch[node_idx], enc_train[node_idx], 'r',
                     g_epoch[node_idx], enc_valid[node_idx], 'r--')
            plt.plot(g_epoch[node_idx], clf_1_train[node_idx], 'b',
                     g_epoch[node_idx], clf_1_valid[node_idx], 'b--')
            plt.plot(g_epoch[node_idx], clf_2_train[node_idx], 'g',
                     g_epoch[node_idx], clf_2_valid[node_idx], 'g--')
            plt.legend([
                'encoder train', 'encoder valid',
                'clf_1 train', 'clf_1 valid',
                'clf_2 train', 'clf_2 valid',
            ])
            plt.title("Node:{}, EIGAN @ {}".format(node_idx, epoch))
            plt.show()

    return encoders

def distributed_3(
        num_nodes,
        phi,
        delta,
        X_trains,
        X_valids,
        y_1_trains,
        y_1_valids,
        y_2_trains,
        y_2_valids,
        y_3_trains,
        y_3_valids,
        input_size,
        hidden_size,
        output_size,
        alpha,
        lr_encd,
        lr_1,
        lr_2,
        lr_3,
        w_1,
        w_2,
        w_3,
        train_loaders,
        n_iter_gan,
        device,
        global_params,
        plot=True,
        debug=True,
        eps2=-np.log(0.5),
        eps3=-np.log(0.5),
):
    encoders = []
    clfs_1 = []
    clfs_2 = []
    clfs_3 = []
    bce_1_losses = []
    bce_2_losses = []
    bce_3_losses = []
    optimizer = torch.optim.Adam
    encd_optimizers = []
    clf_1_optimizers = []
    clf_2_optimizers = []
    clf_3_optimizers = []

    for _ in range(num_nodes):
        encoders.append(Encoder(input_size=input_size, hidden_size=hidden_size,
                                output_size=input_size).to(device))
        clfs_1.append(Discriminator(input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=output_size[_]).to(device))
        clfs_2.append(Discriminator(input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=output_size[_]).to(device))
        clfs_3.append(Discriminator(input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=output_size[_]).to(device))
        

        bce_1_losses.append(torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(w_1[_])).to(device))
        bce_2_losses.append(torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(w_2[_])).to(device))
        bce_3_losses.append(torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(w_3[_])).to(device))

        encd_optimizers.append(optimizer(encoders[_].parameters(), lr=lr_encd))
        clf_1_optimizers.append(optimizer(clfs_1[_].parameters(), lr=lr_1))
        clf_2_optimizers.append(optimizer(clfs_2[_].parameters(), lr=lr_2))
        clf_3_optimizers.append(optimizer(clfs_3[_].parameters(), lr=lr_3))

    if debug:
        print("epoch \t node \t encoder_train \t encoder_valid \t 1_train "
              "\t 1_valid \t 2_train \t 2_valid \t 3_train \t 3_valid"
        )

    g_epoch = defaultdict(list)
    enc_train = defaultdict(list)
    enc_valid = defaultdict(list)
    clf_1_train = defaultdict(list)
    clf_1_valid = defaultdict(list)
    clf_2_train = defaultdict(list)
    clf_2_valid = defaultdict(list)
    clf_3_train = defaultdict(list)
    clf_3_valid = defaultdict(list)

    for _ in encoders:
        _.train()
    for _ in clfs_1:
        _.train()
    for _ in clfs_2:
        _.train()
    for _ in clfs_3:
        _.train()

    d = 0

    weights = np.zeros((num_nodes))
    for idx in range(num_nodes):
        weights[idx] = len(X_trains[idx])
    weights = weights/weights.sum()
    encoders = federated(encoders, weights, global_params, 1, device)

    for epoch in range(n_iter_gan):
        if d == delta and delta != 0 and phi != 0:
            print('Aggregating on epoch: {}...'.format(epoch))
            encoders = federated(encoders, weights, global_params,
                                 phi, device)
            d = 0
        d += 1
        for node_idx in range(num_nodes):
            # others = [idx for idx in range(num_nodes) if idx != node_idx]
            for _, batch in enumerate(train_loaders[node_idx]):
                x = batch[0].to(device)
                y1 = batch[1].to(device)
                y2 = batch[2].to(device)
                y3 = batch[3].to(device)

                x_ = encoders[node_idx](x)
                y1_ = clfs_1[node_idx](x_)
                y2_ = clfs_2[node_idx](x_)
                y3_ = clfs_3[node_idx](x_)

                clf_1_train_loss = bce_1_losses[node_idx](y1_, y1)
                clf_2_train_loss = bce_2_losses[node_idx](y2_, y2)
                clf_3_train_loss = bce_3_losses[node_idx](y3_, y3)
                encd_train_loss = clf_1_train_loss \
                    - alpha * (clf_2_train_loss - eps2) \
                    - alpha * (clf_3_train_loss - eps3)

                encd_optimizers[node_idx].zero_grad()
                encd_train_loss.backward()
                encd_optimizers[node_idx].step()

                x_ = encoders[node_idx](x)
                y1_ = clfs_1[node_idx](x_)
                clf_1_train_loss = bce_1_losses[node_idx](y1_, y1)

                clf_1_optimizers[node_idx].zero_grad()
                clf_1_train_loss.backward()
                clf_1_optimizers[node_idx].step()

                x_ = encoders[node_idx](x)
                y2_ = clfs_2[node_idx](x_)
                clf_2_train_loss = bce_2_losses[node_idx](y2_, y2)

                clf_2_optimizers[node_idx].zero_grad()
                clf_2_train_loss.backward()
                clf_2_optimizers[node_idx].step()

                x_ = encoders[node_idx](x)
                y3_ = clfs_3[node_idx](x_)
                clf_3_train_loss = bce_3_losses[node_idx](y3_, y3)

                clf_3_optimizers[node_idx].zero_grad()
                clf_3_train_loss.backward()
                clf_3_optimizers[node_idx].step()

            X_valid_ = encoders[node_idx](X_valids[node_idx].to(device))
            y_1_valid_ = clfs_1[node_idx](X_valid_)
            y_2_valid_ = clfs_2[node_idx](X_valid_)
            y_3_valid_ = clfs_3[node_idx](X_valid_)

            clf_1_valid_loss = bce_1_losses[node_idx](
                y_1_valid_, y_1_valids[node_idx].to(device))
            clf_2_valid_loss = bce_2_losses[node_idx](
                y_2_valid_, y_2_valids[node_idx].to(device))
            clf_3_valid_loss = bce_3_losses[node_idx](
                y_3_valid_, y_3_valids[node_idx].to(device))
            
            encd_valid_loss = clf_1_valid_loss \
                - alpha * (clf_2_valid_loss - eps2) \
                - alpha * (clf_3_valid_loss - eps3)
            
            if plot:
                g_epoch[node_idx].append(epoch)
                enc_train[node_idx].append(encd_train_loss.item())
                enc_valid[node_idx].append(encd_valid_loss.item())
                clf_1_train[node_idx].append(clf_1_train_loss.item())
                clf_1_valid[node_idx].append(clf_1_valid_loss.item())
                clf_2_train[node_idx].append(clf_2_train_loss.item())
                clf_2_valid[node_idx].append(clf_2_valid_loss.item())
                clf_3_train[node_idx].append(clf_3_train_loss.item())
                clf_3_valid[node_idx].append(clf_3_valid_loss.item())

            if epoch % 20 != 0:
                continue

            if debug:
                print('{} \t {} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} '
                      '\t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f}'.format(
                          epoch, 
                          node_idx,
                          encd_train_loss.item(),
                          encd_valid_loss.item(),
                          clf_1_train_loss.item(),
                          clf_1_valid_loss.item(),
                          clf_2_train_loss.item(),
                          clf_2_valid_loss.item(),
                          clf_3_train_loss.item(),
                          clf_3_valid_loss.item()
                      ))

    if plot:
        for node_idx in range(num_nodes):
            plt.plot(g_epoch[node_idx], enc_train[node_idx], 'r',
                     g_epoch[node_idx], enc_valid[node_idx], 'r--')
            plt.plot(g_epoch[node_idx], clf_1_train[node_idx], 'b',
                     g_epoch[node_idx], clf_1_valid[node_idx], 'b--')
            plt.plot(g_epoch[node_idx], clf_2_train[node_idx], 'g',
                     g_epoch[node_idx], clf_2_valid[node_idx], 'g--')
            plt.plot(g_epoch[node_idx], clf_3_train[node_idx], 'y',
                     g_epoch[node_idx], clf_3_valid[node_idx], 'y--')
            plt.legend([
                'encoder train', 'encoder valid',
                'clf_1 train', 'clf_1 valid',
                'clf_2 train', 'clf_2 valid',
                'clf_3 train', 'clf_3 valid',
            ])
            plt.title("Node:{}, EIGAN @ {}".format(node_idx, epoch))
            plt.show()

    return encoders


def distributed_3_diff(
        num_nodes,
        phi,
        delta,
        X_trains,
        X_valids,
        y_1_trains,
        y_1_valids,
        y_2_trains,
        y_2_valids,
        y_3_trains,
        y_3_valids,
        input_size,
        hidden_size,
        output_size,
        alpha,
        lr_encd,
        lr_1,
        lr_2,
        lr_3,
        w_1,
        w_2,
        w_3,
        train_loaders,
        n_iter_gan,
        device,
        global_params,
        plot=True,
        debug=True,
        eps2=-np.log(0.5),
        eps3=-np.log(0.5),
):
    encoders = []
    clfs_1 = []
    clfs_2 = []
    clfs_3 = []
    bce_1_losses = []
    bce_2_losses = []
    bce_3_losses = []
    optimizer = torch.optim.Adam
    encd_optimizers = []
    clf_1_optimizers = []
    clf_2_optimizers = []
    clf_3_optimizers = []

    clf_2_active = [0, 2, 4, 6, 8]
    clf_3_active = [1, 3, 5, 7, 9]

    for _ in range(num_nodes):
        encoders.append(Encoder(input_size=input_size, hidden_size=hidden_size,
                                output_size=input_size).to(device))
        clfs_1.append(Discriminator(input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=output_size[_]).to(device))
        if _ in clf_2_active:
            clfs_2.append(Discriminator(input_size=input_size,
                                        hidden_size=hidden_size,
                                        output_size=output_size[_]).to(device))
            clf_2_optimizers.append(optimizer(clfs_2[_].parameters(), lr=lr_2))
        else:
            clfs_2.append(None)
            clf_2_optimizers.append(None)
        if _ in clf_3_active:
            clfs_3.append(Discriminator(input_size=input_size,
                                        hidden_size=hidden_size,
                                        output_size=output_size[_]).to(device))
            clf_3_optimizers.append(optimizer(clfs_3[_].parameters(), lr=lr_3))
        else:
            clfs_3.append(None)
            clf_3_optimizers.append(None)

        bce_1_losses.append(torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(w_1[_])).to(device))
        bce_2_losses.append(torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(w_2[_])).to(device))
        bce_3_losses.append(torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor(w_3[_])).to(device))

        encd_optimizers.append(optimizer(encoders[_].parameters(), lr=lr_encd))
        clf_1_optimizers.append(optimizer(clfs_1[_].parameters(), lr=lr_1))

    if debug:
        print("epoch \t node \t encoder_train \t encoder_valid \t 1_train "
              "\t 1_valid \t 2_train \t 2_valid \t 3_train \t 3_valid"
        )

    g_epoch = defaultdict(list)
    enc_train = defaultdict(list)
    enc_valid = defaultdict(list)
    clf_1_train = defaultdict(list)
    clf_1_valid = defaultdict(list)
    clf_2_train = defaultdict(list)
    clf_2_valid = defaultdict(list)
    clf_3_train = defaultdict(list)
    clf_3_valid = defaultdict(list)

    for _ in encoders:
        _.train()
    for _ in clfs_1:
        _.train()
    for _ in clfs_2:
        if _:
            _.train()
    for _ in clfs_3:
        if _:
            _.train()

    d = 0

    weights = np.zeros((num_nodes))
    for idx in range(num_nodes):
        weights[idx] = len(X_trains[idx])
    weights = weights/weights.sum()
    encoders = federated(encoders, weights, global_params, 1, device)

    for epoch in range(n_iter_gan):
        if d == delta and delta != 0 and phi != 0:
            print('Aggregating on epoch: {}...'.format(epoch))
            encoders = federated(encoders, weights, global_params,
                                 phi, device)
            d = 0
        d += 1
        for node_idx in range(num_nodes):
            # others = [idx for idx in range(num_nodes) if idx != node_idx]
            for _, batch in enumerate(train_loaders[node_idx]):
                x = batch[0].to(device)
                y1 = batch[1].to(device)
                y2 = batch[2].to(device)
                y3 = batch[3].to(device)

                x_ = encoders[node_idx](x)
                y1_ = clfs_1[node_idx](x_)
                clf_1_train_loss = bce_1_losses[node_idx](y1_, y1)
                clf_2_train_loss = eps2
                if node_idx in clf_2_active:
                    y2_ = clfs_2[node_idx](x_)
                    clf_2_train_loss = bce_2_losses[node_idx](y2_, y2)
                clf_3_train_loss = eps3
                if node_idx in clf_3_active:
                    y3_ = clfs_3[node_idx](x_)
                    clf_3_train_loss = bce_3_losses[node_idx](y3_, y3)

                encd_train_loss = clf_1_train_loss \
                    - alpha * (clf_2_train_loss - eps2) \
                    - alpha * (clf_3_train_loss - eps3)

                encd_optimizers[node_idx].zero_grad()
                encd_train_loss.backward()
                encd_optimizers[node_idx].step()

                x_ = encoders[node_idx](x)
                y1_ = clfs_1[node_idx](x_)
                clf_1_train_loss = bce_1_losses[node_idx](y1_, y1)

                clf_1_optimizers[node_idx].zero_grad()
                clf_1_train_loss.backward()
                clf_1_optimizers[node_idx].step()

                if node_idx in clf_2_active:
                    x_ = encoders[node_idx](x)
                    y2_ = clfs_2[node_idx](x_)
                    clf_2_train_loss = bce_2_losses[node_idx](y2_, y2)

                    clf_2_optimizers[node_idx].zero_grad()
                    clf_2_train_loss.backward()
                    clf_2_optimizers[node_idx].step()

                if node_idx in clf_3_active:
                    x_ = encoders[node_idx](x)
                    y3_ = clfs_3[node_idx](x_)
                    clf_3_train_loss = bce_3_losses[node_idx](y3_, y3)

                    clf_3_optimizers[node_idx].zero_grad()
                    clf_3_train_loss.backward()
                    clf_3_optimizers[node_idx].step()

            X_valid_ = encoders[node_idx](X_valids[node_idx].to(device))
            y_1_valid_ = clfs_1[node_idx](X_valid_)
            clf_2_valid_loss = eps2
            if node_idx in clf_2_active:
                y_2_valid_ = clfs_2[node_idx](X_valid_)
                clf_2_valid_loss = bce_2_losses[node_idx](
                    y_2_valid_, y_2_valids[node_idx].to(device))
            clf_3_valid_loss = eps3
            if node_idx in clf_3_active:
                y_3_valid_ = clfs_3[node_idx](X_valid_)
                clf_3_valid_loss = bce_3_losses[node_idx](
                    y_3_valid_, y_3_valids[node_idx].to(device))

            clf_1_valid_loss = bce_1_losses[node_idx](
                y_1_valid_, y_1_valids[node_idx].to(device))

            encd_valid_loss = clf_1_valid_loss \
                - alpha * (clf_2_valid_loss - eps2) \
                - alpha * (clf_3_valid_loss - eps3)
            
            if plot:
                g_epoch[node_idx].append(epoch)
                enc_train[node_idx].append(encd_train_loss.item())
                enc_valid[node_idx].append(encd_valid_loss.item())
                clf_1_train[node_idx].append(clf_1_train_loss.item())
                clf_1_valid[node_idx].append(clf_1_valid_loss.item())
                if node_idx in clf_2_active:
                    clf_2_train[node_idx].append(clf_2_train_loss.item())
                    clf_2_valid[node_idx].append(clf_2_valid_loss.item())
                else:
                    clf_2_train[node_idx].append(0)
                    clf_2_valid[node_idx].append(0)
                if node_idx in clf_3_active:
                    clf_3_train[node_idx].append(clf_3_train_loss.item())
                    clf_3_valid[node_idx].append(clf_3_valid_loss.item())
                else:
                    clf_3_train[node_idx].append(0)
                    clf_3_valid[node_idx].append(0)

            if epoch % 20 != 0:
                continue

            if debug:
                print('{} \t {} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} '
                      '\t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f}'.format(
                          epoch, 
                          node_idx,
                          encd_train_loss.item(),
                          encd_valid_loss.item(),
                          clf_1_train[node_idx][-1],
                          clf_1_valid[node_idx][-1],
                          clf_2_train[node_idx][-1],
                          clf_2_valid[node_idx][-1],
                          clf_3_train[node_idx][-1],
                          clf_3_valid[node_idx][-1],
                      ))

    if plot:
        for node_idx in range(num_nodes):
            plt.plot(g_epoch[node_idx], enc_train[node_idx], 'r',
                     g_epoch[node_idx], enc_valid[node_idx], 'r--')
            plt.plot(g_epoch[node_idx], clf_1_train[node_idx], 'b',
                     g_epoch[node_idx], clf_1_valid[node_idx], 'b--')
            plt.plot(g_epoch[node_idx], clf_2_train[node_idx], 'g',
                     g_epoch[node_idx], clf_2_valid[node_idx], 'g--')
            plt.plot(g_epoch[node_idx], clf_3_train[node_idx], 'y',
                     g_epoch[node_idx], clf_3_valid[node_idx], 'y--')
            plt.legend([
                'encoder train', 'encoder valid',
                'clf_1 train', 'clf_1 valid',
                'clf_2 train', 'clf_2 valid',
                'clf_3 train', 'clf_3 valid',
            ])
            plt.title("Node:{}, EIGAN @ {}".format(node_idx, epoch))
            plt.show()

    return encoders
