import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class ConstrainedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', max_weight=1.0):

        super(ConstrainedConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, dilation=dilation, groups=groups,
                                                bias=bias, padding_mode=padding_mode)
        self.max_weight = max_weight
    def forward(self, input):
        return F.conv2d(input, self.weight.clamp(min=-self.max_weight, max=self.max_weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class ConstrainedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, max_weight=1.0):
        super(ConstrainedLinear, self).__init__(in_features=in_features, out_features=out_features,
                                                bias=bias)
        self.max_weight = max_weight

    def forward(self, input):
        return F.linear(input, self.weight.clamp(min=-self.max_weight, max=self.max_weight),
                        self.bias)

class TimeDropout2d(nn.Dropout2d):
    """
    Dropout layer, where the last dimension is treated as channels
    """
    def __init__(self, p=0.5, inplace=False):
        """
        See nn.Dropout2d for parameters
        """
        super(TimeDropout2d, self).__init__(p=p, inplace=inplace)

    def forward(self, input):
        if self.training:
            input = input.permute(0, 3, 1, 2)
            input = F.dropout2d(input, self.p, True, self.inplace)
            input = input.permute(0, 2, 3, 1)
        return input

