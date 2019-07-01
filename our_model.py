import torch.nn as nn
import torch
import torch.nn.functional as F
import math

torch.manual_seed(123)


class Naive_Model(nn.Module):
    def __init__(self):
        super(Naive_Model, self).__init__()
        # self.device = device
        self.h2o = nn.Linear(input_size, number_of_classes)
        self.softmax = nn.LogSoftmax(dim=2)

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
        self.conv1 = nn.Conv2d(1, first_out_channel, filter_one_size, stride=first_filter_stride)
        self.pool1 = nn.MaxPool2d(2, 2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(first_out_channel)
        self.conv2 = nn.Conv2d(first_out_channel, second_filter_channels, second_filter_size,
                               stride=second_filter_stride)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(second_filter_channels)


        # self.conv3 = nn.Conv2d(second_filter_channels, third_filter_channels,third_filter_size, third_filter_stride)
        self.rnn_module = nn.LSTM(dim_into_rnn, hidden_size, num_layers=1, batch_first=True,
                                  bidirectional=bidirectional)
        self.last_layer_bias = torch.autograd.Variable(torch.zeros((number_of_classes,)))
        self.conv2output = nn.Linear(dim_without_rnn, number_of_classes)
        self.h2o = nn.Linear( (1 + bidirectional) * hidden_size, number_of_classes)

        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, tensor_input):
        x = self.conv1(tensor_input)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.conv1_bn(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.conv2_bn(x)
        x_squeeze = torch.squeeze(x, 1)
        x_squeeze = x_squeeze.permute(2, 0, 1)

        packed_output,h_n = self.rnn_module(x_squeeze)
        rnn_output = packed_output.permute(1,0,2)
        predictions = self.h2o(rnn_output)
        # output_from_cnn = self.last_layer_bias()


        output = self.softmax(predictions)
        return output

    def get_seq_length(self,input_length):
        # in_to_second_conv = ((input_length - filter_one_size[1]) // first_filter_stride + 1)
        # return ((in_to_second_conv - second_filter_size[1]) // second_filter_stride + 1)
        return input_to_bias_width


class model_with_pooling(nn.Module):
    def __init__(self):
        super(model_with_pooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 10, stride=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(15)

        self.pooling2d = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(15, 1, 5,stride=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(1)

        self.dropout = nn.Dropout2d(p=0.4)
        self.rnn_module = nn.LSTM(input_size=72, hidden_size=72, num_layers=2, batch_first=True,
                                  bidirectional=bidirectional)
        self.rnn2fc = nn.Linear(72*2, 72*2)
        self.h2o = nn.Linear(72*2, 27)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, tensor_input):
        x = self.conv1(tensor_input)
        # x = self.dropout(x)
        x = self.dropout(x)
        x = self.pooling2d(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.low_dropout(x)
        # x = F.relu(x)
        x = self.conv2_bn(x)
        # x = self.conv3(x)
        x = torch.squeeze(x, 1)
        x = x.permute(0, 2, 1)
        packed_output,h_n = self.rnn_module(x)
        fc_layer = self.rnn2fc(packed_output)
        output_from_cnn = self.h2o(fc_layer)
        output = self.softmax(output_from_cnn)
        return output

    def get_seq_length(self,input_length):
        # in_to_second_conv = ((input_length - filter_one_size[1]) // first_filter_stride + 1)
        # return ((in_to_second_conv - second_filter_size[1]) // second_filter_stride + 1)
        return 42


hidden_size = 128
input_size = 161


seq_input_length = 101
filter_one_size = (5, 5)
first_out_channel = 32
first_filter_stride = (2,2)

second_filter_size = (4, 5)
second_filter_channels = 1
second_filter_stride = (1,1)

input_to_second_conv_time = math.floor(((seq_input_length - filter_one_size[1]) // first_filter_stride[1] + 1)/2)
input_to_second_conv_freq = math.floor(((input_size - filter_one_size[0]) // first_filter_stride[0] + 1)/2)

input_to_third_conv_time = ((input_to_second_conv_time - second_filter_size[1]) // second_filter_stride[1] + 1)
input_to_third_conv_freq = ((input_to_second_conv_freq - second_filter_size[0]) // second_filter_stride[0] + 1)


third_filter_size = (input_to_third_conv_freq, 3)
third_filter_channels = 27
third_filter_stride = (1,1)


input_to_bias_width = ((input_to_third_conv_time - third_filter_size[1]) // third_filter_stride[1] + 1)
input_to_bias_height= ((input_to_third_conv_freq - third_filter_size[0]) // third_filter_stride[0] + 1)

rnn_seq_length = ((input_to_second_conv_time - second_filter_size[1]) // second_filter_stride[1] + 1)
dim_into_rnn = ((input_to_second_conv_freq - second_filter_size[0]) // second_filter_stride[0] + 1) * second_filter_channels

dim_without_rnn = 38

number_of_classes = 27
bidirectional = True

