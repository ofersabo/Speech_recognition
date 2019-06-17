import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn_module = nn.RNN(input_size,hidden_size,batch_first=True,bidirectional=False)
        self.h2o = nn.Linear( hidden_size, output_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        packed_output,h_n = self.rnn_module(input)
        output_from_rnn = self.h2o(packed_output)
        # output = self.i2o(combined)
        output = self.softmax(output_from_rnn)
        return output
