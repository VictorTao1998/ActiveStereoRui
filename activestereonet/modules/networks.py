import torch
import torch.nn as nn
import torch.nn.functional as F

from activestereonet.nn.init import init_uniform, init_bn


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(convbn(in_channel, in_channel, 3, 1, 1, dilation), nn.LeakyReLU(0.2, False))

    def forward(self, x):
        out = self.conv(x)
        out += x

        return out


class LeakyConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, alpha=0.2, bn_momentum=0.1, **kwargs):
        super(LeakyConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=False,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channel, momentum=bn_momentum)
        self.leaky_relu = nn.LeakyReLU(alpha, False)

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)

        return x

    def init_weights(self):
        init_uniform(self.conv)
        init_bn(self.bn)


class LeakyConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, alpha=0.2, bn_momentum=0.1, **kwargs):
        super(LeakyConv3d, self).__init__()

        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=False,
                              **kwargs)
        self.bn = nn.BatchNorm3d(out_channel, momentum=bn_momentum)
        self.leaky_relu = nn.LeakyReLU(alpha, False)

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)

        return x

    def init_weights(self):
        init_uniform(self.conv)
        init_bn(self.bn)


class SiameseTower(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SiameseTower, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
        self.residual_blocks = nn.ModuleList()
        for i in range(3):
            self.residual_blocks.append(ResidualBlock(out_channel))
        self.leaky_conv_blocks = nn.ModuleList()
        for i in range(3):
            self.leaky_conv_blocks.append(LeakyConv2d(out_channel, out_channel, 3, stride=2, padding=1))
        self.out_conv = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        for res in self.residual_blocks:
            x = res(x)

        for leaky_conv in self.leaky_conv_blocks:
            x = leaky_conv(x)

        x = self.out_conv(x)

        return x


class CostVolumeFilter(nn.Module):
    def __init__(self, base_channel):
        super(CostVolumeFilter, self).__init__()
        self.filter = nn.Sequential(
            LeakyConv3d(base_channel, base_channel, 3, 1, 1),
            LeakyConv3d(base_channel, base_channel, 3, 1, 1),
            LeakyConv3d(base_channel, base_channel, 3, 1, 1),
            LeakyConv3d(base_channel, base_channel, 3, 1, 1),
        )
        self.out_conv = nn.Conv3d(base_channel, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.filter(x)
        x = self.out_conv(x)

        return x


class DisparityRefinement(nn.Module):
    def __init__(self, base_channel):
        super(DisparityRefinement, self).__init__()
        self.disp_conv = nn.Sequential(
            LeakyConv2d(1, base_channel, 3, 1, 1),
            ResidualBlock(base_channel),
            ResidualBlock(base_channel, 2)
        )
        self.rgb_conv = nn.Sequential(
            LeakyConv2d(1, base_channel, 3, 1, 1),
            ResidualBlock(base_channel),
            ResidualBlock(base_channel, 2)
        )
        self.global_conv = nn.Sequential(
            ResidualBlock(2 * base_channel, 4),
            ResidualBlock(2 * base_channel, 8),
            ResidualBlock(2 * base_channel),
            ResidualBlock(2 * base_channel),
        )
        self.out_conv = nn.Conv2d(2 * base_channel, 1, 3, 1, 1, bias=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.out_conv.weight)

    def forward(self, disp_map, rgb_map):
        disp_feature = self.disp_conv(disp_map)
        rgb_feature = self.rgb_conv(rgb_map)
        cat_feature = torch.cat([disp_feature, rgb_feature], dim=1)
        cat_feature = self.global_conv(cat_feature)
        disp_res = self.out_conv(cat_feature)

        refined_disp_map = disp_map + disp_res

        return refined_disp_map


class InvalidationNetwork(nn.Module):
    def __init__(self, base_channel):
        super(InvalidationNetwork, self).__init__()

        self.tower_convs = nn.Sequential(
            ResidualBlock(2 * base_channel),
            ResidualBlock(2 * base_channel),
            ResidualBlock(2 * base_channel),
            ResidualBlock(2 * base_channel),
            ResidualBlock(2 * base_channel),
            nn.Conv2d(2 * base_channel, 1, 3, 1, 1, bias=False)
        )

        self.refine_conv = nn.Sequential(
            LeakyConv2d(3, base_channel, 3, 1, 1),
            ResidualBlock(base_channel),
            ResidualBlock(base_channel),
            ResidualBlock(base_channel),
            ResidualBlock(base_channel),
            nn.Conv2d(base_channel, 1, 3, 1, 1, bias=False),
        )

    def forward(self, left_tower, right_tower, full_res_disp, left_ir):
        assert full_res_disp.shape[2:] == left_ir.shape[2:]
        tower = torch.cat([left_tower, right_tower], dim=1)
        coarse_invalid_mask = self.tower_convs(tower)
        upsampled_invalid_mask = F.upsample(coarse_invalid_mask, full_res_disp.shape[2:], mode="bilinear")
        cat_feature = torch.cat([upsampled_invalid_mask, full_res_disp, left_ir], dim=1)
        invalid_mask_res = self.refine_conv(cat_feature)
        refined_invalid_mask = upsampled_invalid_mask + invalid_mask_res
        refined_invalid_mask = F.sigmoid(refined_invalid_mask)
        return refined_invalid_mask
