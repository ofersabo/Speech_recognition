import torch.nn as nn
import torch
import torch.nn.functional as F

torch.manual_seed(123)


class Naive_Model(nn.Module):
    def __init__(self):
        super(Naive_Model, self).__init__()
        # self.device = device
        self.h2o = nn.Linear(input_size, number_of_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, tensor_input):
        x_squeeze = torch.squeeze(tensor_input, 1)
        x_squeeze = x_squeeze.permute(0, 2, 1)
        reduced_size = self.h2o(x_squeeze)
        output = self.softmax(reduced_size)
        return output

    def get_seq_length(self,input_length):
        return input_length


class our_model(nn.Module):
    def __init__(self):
        super(our_model, self).__init__()
        # self.device = device
        self.cont = 0
        self.conv1 = nn.Conv2d(1, first_out_channel, filter_one_size, stride=first_filter_stride)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(first_out_channel)
        self.conv2 = nn.Conv2d(first_out_channel, second_filter_channels, second_filter_size,
                               stride=first_filter_stride)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(second_filter_channels)
        self.conv3 = nn.Conv2d(second_filter_channels, 1, 1, stride=1)
        self.rnn_module = nn.LSTM(dim_into_rnn, hidden_size, num_layers=1, batch_first=True,
                                  bidirectional=bidirectional)
        self.h2o = nn.Linear((1 + bidirectional) * hidden_size, number_of_classes)
        self.conv2output = nn.Linear(dim_without_rnn, number_of_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, tensor_input):
        self.cont += 1
        x = self.conv1(tensor_input)
        x = F.relu(x)
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv3(x)
        x_squeeze = torch.squeeze(x, 1)
        x_squeeze = x_squeeze.permute(0, 2, 1)

        # packed_output,h_n = self.rnn_module(x_squeeze)
        # output_from_rnn = self.h2o(packed_output)
        output_from_cnn = self.conv2output(x_squeeze)

        output = self.softmax(output_from_cnn)
        return output

    def get_seq_length(self,input_length):
        in_to_second_conv = ((input_length - filter_one_size[1]) // first_filter_stride + 1)
        return ((in_to_second_conv - second_filter_size[1]) // second_filter_stride + 1)


hidden_size = 128
input_size = 161
seq_input_length = 101
filter_one_size = (5, 5)
first_out_channel = 32
first_filter_stride = 2

second_filter_size = (5, 5)
second_filter_channels = 32
second_filter_stride = 2

input_to_second_conv_time = ((seq_input_length - filter_one_size[1]) // first_filter_stride + 1)
input_to_second_conv_freq = ((input_size - filter_one_size[0]) // first_filter_stride + 1)

rnn_seq_length = ((input_to_second_conv_time - second_filter_size[1]) // second_filter_stride + 1)
dim_into_rnn = ((input_to_second_conv_freq - second_filter_size[
    0]) // second_filter_stride + 1) * second_filter_channels

dim_without_rnn = 38

number_of_classes = 27
bidirectional = False

