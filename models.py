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
        self.hidden_sz = args.hidden_size
        self.max_size = args.max_length
        self.output_size = output_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_sz)
        self.attention = nn.Linear(2 * self.hidden_sz, self.max_size)
        self.combine = nn.Linear(2 * self.hidden_sz, self.hidden_sz)
        self.rnn = nn.GRU(self.hidden_sz, self.hidden_sz)
        self.out = nn.Linear(self.hidden_sz, self.output_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x).view(1, 1, -1)
        attention_weights = F.softmax(self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        with_attention = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], with_attention[0]), 1)
        output = F.relu(self.combine(output).unsqueeze(0))
        output, hidden = self.rnn(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attention_weights
