import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super(Encoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size)

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
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size)
        self.softmax_layer = nn.LogSoftmax(dim=1)
        self.output_layer = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden_layer, encoder_out):
        x = self.embedding(x).view(1, 1, -1)
        x = F.relu(x)
        output, self.hidden_layer = self.rnn(x, hidden_layer)
        output = self.softmax_layer(self.output_layer(output[0]))
        return output, self.hidden_layer


class AttnDecoderRNN(nn.Module):
    def __init__(self, args, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.output_size = output_size
        self.max_length = args.max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
