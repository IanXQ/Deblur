import torch
import torch.nn as nn
from layers import *


def conv5x5_relu(in_channels, out_channels, stride):
    return conv(in_channels, out_channels, 5, stride, 2, activation_fn=partial(nn.ReLU, inplace=True))


def conv3x3_relu(in_channels, out_channels):
    return conv(in_channels, out_channels, 3, 1, 1, activation_fn=partial(nn.ReLU, inplace=True))


def deconv5x5_relu(in_channels, out_channels, stride, output_padding):
    return deconv(in_channels, out_channels, 5, stride, 2, output_padding=output_padding,
                  activation_fn=partial(nn.ReLU, inplace=True))


def resblock(in_channels):
    return BasicBlock(in_channels, out_channels=in_channels, kernel_size=5, stride=1, use_batchnorm=False,
                      activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=None)



class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = conv5x5_relu(in_channels, out_channels, stride)
        resblock_list = []
        resblock_list.append(SELayer(out_channels))
        for i in range(2):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.conv(x)
        residual = x
        for i in range(3):
            x = self.resblock_stack(x)
        return x + residual


class SeEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = conv5x5_relu(in_channels, out_channels, stride)
        resblock_list = []
        resblock_list.append(SELayer(out_channels))
        for i in range(2):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.conv(x)
        residual = x
        for i in range(3):
            x = self.resblock_stack(x)
        return x + residual


class XSBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv3x3_relu(in_channels, out_channels)
        resblock_list = [SELayer(out_channels)]
        for i in range(3):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.conv(x)
        residual = x
        x = self.resblock_stack(x)
        return x + residual


class XSeBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(XSBlock(in_channels,in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        residual = x
        x = self.resblock_stack(x)
        return x + residual


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super(type(self), self).__init__()
        resblock_list = []
        resblock_list.append(SELayer(in_channels))
        for i in range(2):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.deconv = deconv5x5_relu(in_channels, out_channels, stride, output_padding)

    def forward(self, x):
        residual = x
        for i in range(3):
            x = self.resblock_stack(x)
        x = self.deconv(x + residual)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.conv = conv(in_channels, 3, 5, 1, 2, activation_fn=None)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.conv(x)
        return x


class SRNDeblurNet(nn.Module):
    """SRN-DeblurNet
    examples:
        net = SRNDeblurNet()
        y = net( x1 , x2 , x3）#x3 is the coarsest image while x1 is the finest image
    """

    def __init__(self, upsample_fn=partial(torch.nn.functional.interpolate, align_corners=False, mode='bilinear'),
                 xavier_init_all=True):
        super(type(self), self).__init__()
        self.upsample_fn = upsample_fn
        self.inblock = EBlock(3 + 3, 32, 1)
        self.eblock1 = SeEBlock(32, 64, 2)
        self.eblock2 = SeEBlock(64, 128, 2)

        self.denseblock = XSeBlock(128)
        self.dblock1 = DBlock(128, 64, 2, 1)
        self.dblock2 = DBlock(64, 32, 2, 1)
        self.outblock = OutBlock(32)

        self.input_padding = None
        if xavier_init_all:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    torch.nn.init.xavier_normal_(m.weight)
                    # print(name)

    def forward_step(self, x):
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        # h, c = self.convlstm(e128, hidden_state)
        h = self.denseblock(e128)
        d64 = self.dblock1(h)
        d32 = self.dblock2(d64 + e64)
        d3 = self.outblock(d32 + e32)
        return d3

    def forward(self, b1, b2, b3):
        if self.input_padding is None or self.input_padding.shape != b3.shape:
            self.input_padding = torch.zeros_like(b3)
        # h, c = self.convlstm.init_hidden(b3.shape[0], (b3.shape[-2] // 4, b3.shape[-1] // 4))

        i3 = self.forward_step(torch.cat([b3, self.input_padding], 1))
        i3 = i3.clamp(-1, 1)

        i2 = self.forward_step(torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1))
        i2 = i2.clamp(-1, 1)

        i1 = self.forward_step(torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1))
        i1 = i1.clamp(-1, 1)
        # y2 = self.upsample_fn( y1 , (128,128) )
        # y3 = self.upsample_fn( y2 , (64,64) )

        # return y1 , y2 , y3
        return i1, i2, i3

net = SRNDeblurNet()
print(sum(param.numel() for param in net.parameters()))