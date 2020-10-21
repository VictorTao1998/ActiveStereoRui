import torch
import torch.nn as nn
import torch.nn.functional as F

from activestereonet.nn.conv import Conv2d


def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation),
        nn.BatchNorm2d(out_channel))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride, pad, dilation, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(Conv2d(in_channel, out_channel, 3, stride, relu=False, padding=pad, dilation=dilation),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = Conv2d(out_channel, out_channel, 3, 1, relu=False, padding=pad, dilation=dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class EdgeAwareRefinement(nn.Module):
    def __init__(self, in_channel, out_channel, astrous_list=(1, 2, 4, 8, 1, 1)):
        super(EdgeAwareRefinement, self).__init__()
        self.conv2d_feature = nn.Conv2d
        self.resitual_astrous_blocks = nn.ModuleList()
        for di in astrous_list:
            self.resitual_astrous_blocks.append(
                BasicBlock(out_channel, out_channel, 1, di, di)
            )

