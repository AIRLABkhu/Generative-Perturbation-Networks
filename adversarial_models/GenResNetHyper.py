import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Define a resnet block
class ZeroPadBlockHyper(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ZeroPadBlockHyper, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=1, stride=stride)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += F.pad(self.shortcut(x), (0, 0, 0, 0, 0, out.size()[1] - x.size()[1]), 'constant', 0)
        out = F.relu(out)
        return out

class GenResNetHyper(nn.Module):
    def __init__(self, in_channel, in_height, in_width):
        super(GenResNetHyper, self).__init__()

        input_nc = in_height
        output_nc = in_height

        ngf = 64
        n_downsampling = 2
        n_blocks = 3

        norm_type = 'batch'
        act_type = 'relu'
        padding_type = 'reflect'
        use_dropout = False
        res_block = 2


        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        use_bias = norm_type == 'instance'

        self.conv_pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channel, ngf, kernel_size=7, padding=0,
                            bias=use_bias)
        self.conv_norm1 = nn.BatchNorm2d(ngf)
        self.conv_act1 = nn.ReLU(True)

        depth = ngf
        self.conv2_1 = nn.Conv2d(depth,  depth * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias)
        self.conv_norm2_1 = nn.BatchNorm2d(depth * 2)
        self.conv_act2_1 = nn.ReLU(True)

        # depth = depth * 2

        self.conv2_2 = nn.Conv2d(depth, depth * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias)
        self.conv_norm2_2 = nn.BatchNorm2d(depth * 2)
        self.conv_act2_2 = nn.ReLU(True)

        depth = depth * 2

        self.res_layer1 = self._make_layer(ZeroPadBlockHyper, depth, res_block, stride=1)
        # self.res_layer2 = self._make_layer(ZeroPadBlock, depth, res_block, stride=1)
        # self.res_layer3 = self._make_layer(ZeroPadBlock, depth, res_block, stride=1)

        self.deconv1 = nn.ConvTranspose2d(depth, int(depth / 2), kernel_size=3, stride=2, padding=1, output_padding=1,
                                          bias=use_bias)
        self.deconv_norm1 = nn.BatchNorm2d(int(depth / 2))
        self.deconv_act1 = nn.ReLU(True)

        # original -----------------------------
        depth = int(depth/2)

        # self.deconv2 = nn.ConvTranspose2d(depth, int(depth / 2), kernel_size=3, stride=2, padding=1, output_padding=1,
        #                                   bias=use_bias)
        # self.deconv_norm2 = nn.BatchNorm2d(int(depth / 2))
        # self.deconv_act2 = nn.ReLU(True)
        #
        # depth = int(depth / 2)
        # original ----------------------------- end




        self.conv_pad3 = nn.ReflectionPad2d(3)
        self.conv3_1 = nn.Conv2d(depth, 11, kernel_size = 7, padding = 0)

        self.conv3_2 = nn.Conv2d(11, in_channel, kernel_size=1, padding=0)
        self.conv_act3 = nn.Tanh()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, emb):

        #print('-begin-')
        #print(x.size())
        x = self.conv_pad1(x)
        x = self.conv1(x)
        x = self.conv_norm1(x)
        x = self.conv_act1(x)
        #print('-after conv_pad1-')
        #print(x.size())

        x = self.conv2_1(x)
        x = self.conv_norm2_1(x)
        x = self.conv_act2_1(x)

        #print('-after conv2_1-')
        #print(x.size())

        # x = self.conv2_2(x)
        # x = self.conv_norm2_2(x)
        # x = self.conv_act2_2(x)
        #
        # #print('-after conv2_2-')
        # #print(x.size())

        x = self.res_layer1(x)

        #print('-after res_layer1-')
        #print(x.size())

        # x = self.res_layer2(x)

        #print('-after res_layer2-')
        #print(x.size())

        # x = self.res_layer3(x)

        #print('-after res_layer3-')
        #print(x.size())

        x = self.deconv1(x)
        x = self.deconv_norm1(x)
        x = self.deconv_act1(x)

        # #print('-after deconv1-')
        # #print(x.size())
        #
        # x = self.deconv2(x)
        # x = self.deconv_norm2(x)
        # x = self.deconv_act2(x)

        #print('-after deconv2-')
        #print(x.size())

        x = self.conv_pad3(x)
        x = self.conv3_1(x)
        x = x * emb
        x = self.conv3_2(x)
        x = self.conv_act3(x)

        #print('-after conv_pad3-')
        #print(x.size())

        return x

