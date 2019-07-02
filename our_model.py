import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(123)

class SR_model(nn.Module):
    def __init__(self):
        super(SR_model, self).__init__()
        self.conv1 = nn.Conv2d(1, first_out_channel, filter_one_size, stride=first_filter_stride)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(first_out_channel)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.pooling2d = nn.MaxPool2d(filter_one_pooling)
        self.conv2 = nn.Conv2d(first_out_channel, second_filter_channels, second_filter_size,stride= second_filter_stride)
        self.conv2_bn = nn.BatchNorm2d(second_filter_channels)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.rnn_module = nn.LSTM(input_size=dim_into_rnn, hidden_size=rnn_hidden_size, num_layers=2, batch_first=True,
                                  bidirectional=bidirectional)
        bidirection_factor = (1 + bidirectional)
        self.rnn2fc = nn.Linear(rnn_hidden_size * bidirection_factor, rnn_hidden_size * bidirection_factor,)
        self.h2o = nn.Linear(rnn_hidden_size * bidirection_factor, number_of_classes)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, tensor_input):
        x = self.conv1(tensor_input)
        x = self.dropout(x)
        x = self.pooling2d(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = torch.squeeze(x, 1)
        x = x.permute(0, 2, 1)
        packed_output,h_n = self.rnn_module(x)
        fc_layer = self.rnn2fc(packed_output)
        output_from_cnn = self.h2o(fc_layer)
        # output_from_cnn = self.h2o(packed_output)

        output = self.softmax(output_from_cnn)
        return output

    def get_seq_length(self,input_length):

        return rnn_seq_length


number_of_classes = 27
input_size = 161
seq_input_length = 101


first_out_channel = 15
filter_one_size = 10
first_filter_stride = 1
filter_one_pooling = 2

dropout_rate = 0.4

second_filter_size = 5
second_filter_channels = 1
second_filter_stride = 1
second_filter_pooling = 1

input_to_second_conv_time = math.floor(((seq_input_length - filter_one_size) // first_filter_stride + 1)/2)
input_to_second_conv_freq = math.floor(((input_size - filter_one_size) // first_filter_stride + 1)/2)

rnn_hidden_size = 128
rnn_seq_length = ((input_to_second_conv_time - second_filter_size) // second_filter_stride + 1)
dim_into_rnn = ((input_to_second_conv_freq - second_filter_size) // second_filter_stride + 1) * second_filter_channels

bidirectional = True

