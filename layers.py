# wrappers for convenience
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch.nn.init import xavier_normal_, kaiming_normal_
import math
from functools import partial


def get_weight_init_fn(activation_fn):
    """get weight_initialization function according to activation_fn
    Notes
    -------------------------------------
    if activation_fn requires arguments, use partial() to wrap activation_fn
    """
    fn = activation_fn
    if hasattr(activation_fn, 'func'):
        fn = activation_fn.func

    if fn == nn.LeakyReLU:
        negative_slope = 0
        if hasattr(activation_fn, 'keywords'):
            if activation_fn.keywords.get('negative_slope') is not None:
                negative_slope = activation_fn.keywords['negative_slope']
        if hasattr(activation_fn, 'args'):
            if len(activation_fn.args) > 0:
                negative_slope = activation_fn.args[0]
        return partial(kaiming_normal_, a=negative_slope)
    elif fn == nn.ReLU or fn == nn.PReLU:
        return partial(kaiming_normal_, a=0)
    else:
        return xavier_normal_
    return


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fn=None, use_batchnorm=False,
         pre_activation=False, bias=True, weight_init_fn=None):
    """pytorch torch.nn.Conv2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        conv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation:

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    try:
        weight_init_fn(conv.weight)
    except:
        print(conv.weight)
    layers.append(conv)
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


def deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, activation_fn=None,
           use_batchnorm=False, pre_activation=False, bias=True, weight_init_fn=None):
    """pytorch torch.nn.ConvTranspose2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        deconv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))

    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    weight_init_fn(deconv.weight)
    layers.append(deconv)
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


def linear(in_channels, out_channels, activation_fn=None, use_batchnorm=False, pre_activation=False, bias=True,
           weight_init_fn=None):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        linear(3,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    linear = nn.Linear(in_channels, out_channels)
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    weight_init_fn(linear.weight)

    layers.append(linear)
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    use partial() to wrap activation_fn if arguments are needed 
    examples:
        BasicBlock(32,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 , inplace = True ))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_batchnorm=False,
                 activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=partial(nn.ReLU, inplace=True),
                 pre_activation=False, scaling_factor=1.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size, stride, kernel_size // 2, activation_fn,
                          use_batchnorm)
        self.conv2 = conv(out_channels, out_channels, kernel_size, 1, kernel_size // 2, None, use_batchnorm,
                          weight_init_fn=get_weight_init_fn(last_activation_fn))
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = conv(in_channels, out_channels, 1, stride, 0, None, use_batchnorm)
        if last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, drop_rate=0.0):
        super(_DenseLayer, self).__init__()
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        self.add_module("selayer", SELayer(channel=num_input_features)),
        self.add_module('conv1',
                        nn.Conv2d(num_input_features, num_input_features, kernel_size=3, stride=1, padding=1,
                                  bias=True)),
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(num_input_features, num_input_features,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return x + new_features


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
