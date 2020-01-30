import torch
import torch.nn as nn


class GeneratorCNN(nn.Module):
    def __init__(self, ngpu, nc, ndf, output_dim):
        super(GeneratorCNN, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 2, 4, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 2, 4, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 2, 4, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, output_dim, 2, 4, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).squeeze(2).squeeze(2)


class GeneratorFCN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size,
        leaky=False, activation='sigmoid', dropout=0.5
    ):
        super(GeneratorFCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.leaky = leaky
        self.dropout = dropout
        self.activation = activation
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=self.dropout)

    def forward(self, x):
        hidden = self.fc1(x)
        if self.leaky:
            relu = self.leaky_relu(hidden)
        else:
            relu = self.relu(hidden)
        # relu = self.dropout(relu)
        output = self.fc2(relu)
        if self.activation == 'sigmoid':
            output = self.sigmoid(output)
        elif self.activation == 'tanh':
            output = self.tanh(output)

        return output


class DiscriminatorFCN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size,
        leaky=False, activation='sigmoid', dropout=0.5
    ):
        super(DiscriminatorFCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.leaky = leaky
        self.dropout = dropout
        self.activation = activation
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=self.dropout)

    def forward(self, x):
        hidden = self.fc1(x)
        if self.leaky:
            relu = self.leaky_relu(hidden)
        else:
            relu = self.relu(hidden)
        # relu = self.dropout(relu)
        output = self.fc2(relu)
        if self.activation == 'sigmoid':
            output = self.sigmoid(output)
        elif self.activation == 'tanh':
            output = self.tanh(output)

        return output
