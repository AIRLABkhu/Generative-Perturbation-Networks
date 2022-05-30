import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import *

class ShallowConvNet(nn.Module):
    def __init__(self, num_channel, num_length, num_label=2):
        super(ShallowConvNet, self).__init__()

        self.conv_filter = 13
        self.strides = 7
        self.pool_size = 32

        self.n_filters_time = 40
        self.n_filters_spatial = 40
        self.dropout_rate = 0.5

        self.n_features = math.ceil((num_length - self.pool_size) / self.strides)

        print(type(self.n_features))

        # self.conv_time = nn.Conv2d(1, self.n_filters_time,  (1, self.conv_filter), stride=1)
        self.sep_conv_pad = nn.ZeroPad2d(((int)(self.conv_filter/2), (int)(self.conv_filter/2), 0, 0))
        self.conv_time = ConstrainedConv2d(1, self.n_filters_time,  (1, self.conv_filter), stride=1, max_weight=2.0)
        self.conv_spatial = ConstrainedConv2d(self.n_filters_time, self.n_filters_spatial,  (num_channel, 1), stride=1, max_weight=2.0)
        self.batch_norm1 = nn.BatchNorm2d(self.n_filters_spatial, momentum=0.01, eps=0.001)
        self.pool1 = nn.AvgPool2d((1, self.pool_size), stride = (1, self.strides))
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.flatten = nn.Flatten()
        self.fc = ConstrainedLinear(self.n_filters_spatial * self.n_features , num_label, bias=True, max_weight=0.5)

        print(self.n_filters_spatial)
        print(self.n_features)
        print(self.n_filters_spatial * self.n_filters_spatial)

        self.initialize_params()

    def forward(self, x):

        # Block 1
        x = self.sep_conv_pad(x)
        x = self.conv_time(x)
        x = self.conv_spatial(x)
        x = self.batch_norm1(x)
        x = torch.square(x)
        x = self.pool1(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout1(x)

        # Classification
        x = self.flatten(x)
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



