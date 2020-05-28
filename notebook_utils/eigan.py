import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size*2)
        self.fc3 = torch.nn.Linear(self.hidden_size*2, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden = self.fc2(relu)
        relu = self.relu(hidden)
        output = self.fc3(relu)
        output = self.tanh(output)
        return output


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size*2)
        self.fc3 = torch.nn.Linear(self.hidden_size*2, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden = self.fc2(relu)
        relu = self.relu(hidden)
        output = self.fc3(relu)
        return output
