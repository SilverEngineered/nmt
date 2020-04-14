import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super(Encoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)

    def forward(self, x, hidden):
        x = self.embedding(x).view(1, 1, -1)
        output, hidden_layer = self.rnn(x, hidden)
        return output, hidden_layer

    def Hidden(self):
        torch.zeros(1, 1, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, args, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        self.softmax_layer = nn.LogSoftmax(dim=1)
        self.output_layer = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden_layer, encoder_out):
        x = self.embedding(x).view(1, 1, -1)
        x = F.relu(x)
        output, self.hidden_layer = self.rnn(x, hidden_layer)
        output = self.softmax_layer(self.output_layer(output[0]))
        return output, self.hidden_layer

