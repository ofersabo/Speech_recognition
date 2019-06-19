import torch.nn as nn
import torch
import torch.nn.functional as F
torch.manual_seed(1)

class our_model(nn.Module):
    def __init__(self):
        super(our_model, self).__init__()
        # self.device = device
        self.cont = 0
        self.conv1 = nn.Conv2d(1, first_out_channel , filter_one_size,stride=first_filter_stride)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(first_out_channel, second_filter_channels , second_filter_size,stride=first_filter_stride)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.rnn_module = nn.LSTM(dim_into_rnn , hidden_size,num_layers= 3,batch_first=True,bidirectional=bidirectional)
        self.h2o = nn.Linear( (1 + bidirectional) * hidden_size, number_of_classes)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, tensor_input):
        self.cont += 1
        x = self.conv1(tensor_input)
        x = F.relu(x)
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = x.view(x.size(0), rnn_seq_length, dim_into_rnn)

        packed_output,h_n = self.rnn_module(x)
        output_from_rnn = self.h2o(packed_output)
        # output = self.i2o(combined)
        output = self.softmax(output_from_rnn)
        return output


hidden_size = 128
input_size = 161
seq_input_length = 101
filter_one_size = (41,11)
first_out_channel = 32
first_filter_stride = 2

second_filter_size = (21,11)
second_filter_channels = 32
second_filter_stride = 2

input_to_second_conv_time = ((seq_input_length - filter_one_size[1]) // first_filter_stride + 1)
input_to_second_conv_freq = ((input_size - filter_one_size[0])//first_filter_stride + 1)


rnn_seq_length = ((input_to_second_conv_time - second_filter_size[1]) // second_filter_stride + 1)
dim_into_rnn = ((input_to_second_conv_freq - second_filter_size[0])//second_filter_stride + 1) * second_filter_channels

number_of_classes = 27
bidirectional = True
#
def get_seq_length(input_length):
    in_to_second_conv = ((input_length - filter_one_size[1]) // first_filter_stride + 1)
    return ((in_to_second_conv - second_filter_size[1]) // second_filter_stride + 1)