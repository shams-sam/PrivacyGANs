import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from .eigan import Encoder, Discriminator
from .utility import class_plot


def bertran(
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
        train_loader,
        n_iter_gan,
        device,
        plot=True,
        debug=True,
        eps=0.2):
    encoder = Encoder(input_size=input_size, hidden_size=hidden_size,
                      output_size=output_size).to(device)
    clf_1 = Discriminator(input_size=input_size, hidden_size=hidden_size,
                          output_size=output_size).to(device)
    clf_2 = Discriminator(input_size=input_size, hidden_size=hidden_size,
                          output_size=output_size).to(device)

    kldiv = torch.nn.KLDivLoss().to(device)
    optimizer = torch.optim.Adam
    encd_optimizer = optimizer(encoder.parameters(), lr=lr_encd)
    clf_1_optimizer = optimizer(clf_1.parameters(), lr=lr_1)
    clf_2_optimizer = optimizer(clf_2.parameters(), lr=lr_2)

    if debug:
        print("epoch \t encoder_train \t encoder_valid \t 1_train "
              "\t 1_valid \t 2_train \t 2_valid"
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

            clf_1_train_loss = kldiv(y1_, y1)
            clf_2_train_loss = kldiv(y2_, y2)
            encd_train_loss = clf_1_train_loss \
                + alpha * torch.square(clf_2_train_loss - eps)

            encd_optimizer.zero_grad()
            encd_train_loss.backward()
            encd_optimizer.step()

            x_ = encoder(x)
            y1_ = clf_1(x_)
            clf_1_train_loss = kldiv(y1_, y1)

            clf_1_optimizer.zero_grad()
            clf_1_train_loss.backward()
            clf_1_optimizer.step()

            x_ = encoder(x)
            y2_ = clf_2(x_)
            clf_2_train_loss = kldiv(y2_, y2)

            clf_2_optimizer.zero_grad()
            clf_2_train_loss.backward()
            clf_2_optimizer.step()

        X_valid_ = encoder(X_valid)
        y_1_valid_ = clf_1(X_valid_)
        y_2_valid_ = clf_2(X_valid_)

        clf_1_valid_loss = kldiv(y_1_valid_, y_1_valid)
        clf_2_valid_loss = kldiv(y_2_valid_, y_2_valid)
        encd_valid_loss = clf_1_valid_loss \
            + alpha * torch.square(clf_2_valid_loss - eps)

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
            print('{} \t {:.8f} \t {:.8f} \t {:.8f} \t {:.8f} '
                  '\t {:.8f} \t {:.8f}'.format(
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
