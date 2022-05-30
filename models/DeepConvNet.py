import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

class DeepConvNet(nn.Module):
    def __init__(self, num_channel, num_length, num_label=2):
        super(DeepConvNet, self).__init__()

        self.conv_filter = 5
        self.stride = 2
        self.pool_size = 2

        self.n_filters_time1 = 25
        self.n_filters_time2= 50
        self.n_filters_time3 = 100
        self.n_filters_time4 = 200
        self.n_filters_spatial = 25
        self.dropout_rate = 0.25

        n_features = num_length // (self.pool_size**4)

        self.conv_pad1 = nn.ZeroPad2d(((int)(self.conv_filter/2), (int)(self.conv_filter/2), 0, 0))
        self.conv_time1 = ConstrainedConv2d(1, self.n_filters_time1,  (1, self.conv_filter), stride=1, max_weight=2.0)
        self.conv_spatial1 = ConstrainedConv2d(self.n_filters_time1, self.n_filters_spatial,  (num_channel, 1), stride=1, max_weight=2.0)
        self.batch_norm1 = nn.BatchNorm2d(self.n_filters_spatial, momentum=0.1, eps=1e-5)
        self.activation1 = nn.ELU(inplace=True)
        self.pool1 = nn.MaxPool2d((1, self.pool_size), stride=(1, self.stride))
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        self.conv_pad2 = nn.ZeroPad2d(((int)(self.conv_filter / 2), (int)(self.conv_filter / 2), 0, 0))
        self.conv_time2 = ConstrainedConv2d(self.n_filters_spatial, self.n_filters_time2, (1, self.conv_filter), stride=1, max_weight=2.0)
        self.batch_norm2 = nn.BatchNorm2d(self.n_filters_time2, momentum=0.1, eps=1e-5)
        self.activation2 = nn.ELU(inplace=True)
        self.pool2 = nn.MaxPool2d((1, self.pool_size), stride=(1, self.stride))
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

        self.conv_pad3 = nn.ZeroPad2d(((int)(self.conv_filter / 2), (int)(self.conv_filter / 2), 0, 0))
        self.conv_time3 = ConstrainedConv2d(self.n_filters_time2, self.n_filters_time3, (1, self.conv_filter), stride=1, max_weight=2.0)
        self.batch_norm3 = nn.BatchNorm2d(self.n_filters_time3, momentum=0.1, eps=1e-5)
        self.activation3 = nn.ELU(inplace=True)
        self.pool3 = nn.MaxPool2d((1, self.pool_size), stride=(1, self.stride))
        self.dropout3 = nn.Dropout(p=self.dropout_rate)

        self.conv_pad4 = nn.ZeroPad2d(((int)(self.conv_filter / 2), (int)(self.conv_filter / 2), 0, 0))
        self.conv_time4 = ConstrainedConv2d(self.n_filters_time3, self.n_filters_time4, (1, self.conv_filter), stride=1,
                                            max_weight=2.0)
        self.batch_norm4 = nn.BatchNorm2d(self.n_filters_time4, momentum=0.1, eps=1e-5)
        self.activation4 = nn.ELU(inplace=True)
        self.pool4 = nn.MaxPool2d((1, self.pool_size), stride=(1, self.stride))
        self.dropout4 = nn.Dropout(p=self.dropout_rate)

        self.flatten = nn.Flatten()
        self.fc = ConstrainedLinear(self.n_filters_time4 * n_features , num_label, bias=True, max_weight=0.5)

        # self.initialize_params()

    def forward(self, x):

        #print('input')
        #print(x.size())

        # Block 1
        #print()
        x = self.conv_pad1(x)
        x = self.conv_time1(x)
        x = self.conv_spatial1(x)

        #print('1 - conv')
        #print(x.size())

        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.pool1(x)

        #print('1 - pool')
        #print(x.size())

        x = self.dropout1(x)

        # Block 2
        x = self.conv_pad2(x)
        x = self.conv_time2(x)

        #print('2 - conv')
        #print(x.size())

        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.pool2(x)

        #print('2 - pool')
        #print(x.size())

        x = self.dropout2(x)

        # Block 3
        x = self.conv_pad3(x)
        x = self.conv_time3(x)

        #print('3 - conv')
        #print(x.size())

        x = self.batch_norm3(x)
        x = self.activation3(x)
        x = self.pool3(x)

        #print('3 - pool')
        #print(x.size())

        x = self.dropout3(x)

        # Block 4
        x = self.conv_pad4(x)
        x = self.conv_time4(x)

        #print('4 - conv')
        #print(x.size())

        x = self.batch_norm4(x)
        x = self.activation4(x)
        x = self.pool4(x)

        #print('4 - pool')
        #print(x.size())

        x = self.dropout4(x)

        # Classification
        x = self.flatten(x)

        #print('fc - flatten')
        #print(x.size())

        x = self.fc(x)
        return x

    def initialize_params(self, weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_,
                          weight_gain=None, bias_gain=None):
        """
        Initializes all the parameters of the model
        Parameters:
         - weight_init: nn.init inplace function
         - bias_init:   nn.init inplace function
         - weight_gain: float, if None, don't use gain for weights
         - bias_gain:   float, if None, don't use gain for bias
        """
        # use gain only if xavier_uniform or xavier_normal is used
        weight_params = {}
        bias_params = {}
        if weight_gain is not None:
            weight_params['gain'] = weight_gain
        if bias_gain is not None:
            bias_params['gain'] = bias_gain

        def init_weight(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight_init(m.weight, **weight_params)
            if isinstance(m, nn.Linear):
                bias_init(m.bias, **bias_params)

        self.apply(init_weight)
