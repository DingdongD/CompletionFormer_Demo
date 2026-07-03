import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_relu(ch_in, ch_out, kernel=3, stride=1, padding=1, relu=True):
    return ConvReLU(ch_in, ch_out, kernel=kernel, stride=stride, padding=padding, relu=relu)


class ConvReLU(nn.Module):
    def __init__(self, ch_in, ch_out, kernel=3, stride=1, padding=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel, stride=stride, padding=padding, bias=True)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        return x


class ResizeConvBasicBlockNoCBAM(nn.Module):
    """Training-side equivalent of the RHB decoder subgraph.

    This replaces ConvTranspose2d + BN + CBAM BasicBlock with
    bilinear resize + Conv/ReLU + two Conv residual block. This is the
    structure currently validated by the compiler-aligned decoder tests.
    """

    def __init__(self, ch_in, ch_out, out_hw):
        super().__init__()
        self.out_hw = out_hw
        self.up_conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=True)
        self.block_conv0 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=True)
        self.block_conv1 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        x = F.relu(self.up_conv(x))
        out = F.relu(self.block_conv0(x))
        out = self.block_conv1(out)
        return F.relu(out + x)


class ConvHead(nn.Module):
    def __init__(self, ch_in, ch_out, relu=False, sigmoid=False):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=True)
        self.relu = relu
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x

