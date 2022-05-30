import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

class PermutedFlatten(nn.Flatten):
    """
    Flattens the input vector in the same way as Keras does
    """
    def __init__(self, start_dim=1, end_dim=-1):
        super(PermutedFlatten, self).__init__(start_dim, end_dim)

    def forward(self, input):
        return input.permute(0, 2, 3, 1).flatten(self.start_dim, self.end_dim)

class EEGNet(nn.Module):
    def __init__(self, num_channel, num_length, num_label=2):
        super(EEGNet, self).__init__()

        """
                F1:           Number of spectral filters
                D:            Number of spacial filters (per spectral filter), F2 = F1 * D
                F2:           Number or None. If None, then F2 = F1 * D
                C:            Number of EEG channels
                T:            Number of time samples
                N:            Number of classes
                p_dropout:    Dropout Probability
                reg_rate:     Regularization (L1) of the final linear layer (fc)
                              This parameter is ignored when constrain_w is not asserted
                activation:   string, either 'elu' or 'relu'
                constrain_w:  bool, if True, constrain weights of spatial convolution and final fc-layer
                dropout_type: string, either 'dropout', 'SpatialDropout2d' or 'TimeDropout2D'
                permuted_flatten: bool, if True, use the permuted flatten to make the model keras compliant
                """
        F1 = 8
        D = 2
        F2 = 16
        C = num_channel
        T = num_length
        N = num_label
        p_dropout = 0.5
        reg_rate = 0.25
        activation = 'relu'
        constrain_w = True
        dropout_type = 'dropout'
        permuted_flatten = False
        avgpool_size1 = 4
        avgpool_size2 = 4
        kernel_size = 64

        # Make the model complicated, for avoidance of underfitting
        F1 = 64
        D = 2
        F2 = 128
        kernel_size = 32

        # check the activation input
        activation = activation.lower()
        assert activation in ['elu', 'relu']

        # Prepare Dropout Type
        if dropout_type.lower() == 'dropout':
            dropout = nn.Dropout
        elif dropout_type.lower() == 'spatialdropout2d':
            dropout = nn.Dropout2d
        elif dropout_type.lower() == 'timedropout2d':
            dropout = TimeDropout2d
        else:
            raise ValueError("dropout_type must be one of SpatialDropout2d, Dropout or "
                             "WrongDropout2d")

        # store local values
        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.p_dropout, self.reg_rate, self.activation = (p_dropout, reg_rate, activation)
        self.constrain_w, self.dropout_type = (constrain_w, dropout_type)

        # Number of input neurons to the final fully connected layer
        n_features = (T // avgpool_size1) // avgpool_size2

        # Block 1
        # self.conv1_pad = nn.ZeroPad2d((31, 32, 0, 0))
        self.conv1_pad = nn.ZeroPad2d(((int)(kernel_size/2 -1), (int)(kernel_size/2), 0, 0))
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_size), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        if constrain_w:
            self.conv2 = ConstrainedConv2d(F1, D * F1, (C, 1), groups=F1, bias=False,
                                           max_weight=1.0)
        else:
            self.conv2 = nn.Conv2d(F1, D * F1, (C, 1), groups=F1, bias=False)

        self.batch_norm2 = nn.BatchNorm2d(D * F1, momentum=0.01, eps=0.001)
        self.activation1 = nn.ELU(inplace=True) if activation == 'elu' else nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d((1, avgpool_size1))
        self.dropout1 = nn.Dropout(p=p_dropout)

        # Block 2
        # Separable Convolution (as described in the paper) is a depthwise convolution followed by
        # a pointwise convolution.
        self.sep_conv_pad = nn.ZeroPad2d((7, 8, 0, 0))
        self.sep_conv1 = nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, bias=False)
        self.sep_conv2 = nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = nn.ELU(inplace=True) if activation == 'elu' else nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d((1, avgpool_size2))
        self.dropout2 = dropout(p=p_dropout)

        # Fully connected layer (classifier)
        if permuted_flatten:
            self.flatten = PermutedFlatten()
        else:
            self.flatten = nn.Flatten()

        if constrain_w:
            self.fc = ConstrainedLinear(F2 * n_features, N, bias=True, max_weight=reg_rate)
        else:
            self.fc = nn.Linear(F2 * n_features, N, bias=True)

        self.initialize_params()

    def forward(self, x):

        # Block 1
        x = self.conv1_pad(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

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

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

