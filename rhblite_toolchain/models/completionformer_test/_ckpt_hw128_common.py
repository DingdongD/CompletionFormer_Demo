import os
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


CKPT_PATH = os.environ.get(
    "COMPLETIONFORMER_HW_CKPT",
    "/root/demo/CompletionFormer/ref_model_hw/model_00030.pt",
)


@lru_cache(maxsize=1)
def ckpt_net():
    return torch.load(CKPT_PATH, map_location="cpu", weights_only=False)["net"]


def copy_conv(conv, prefix, in_slice=None):
    net = ckpt_net()
    weight = net[f"{prefix}.weight"]
    if in_slice is not None:
        weight = weight[:, in_slice]
    with torch.no_grad():
        conv.weight.copy_(weight)
        if conv.bias is not None:
            conv.bias.copy_(net[f"{prefix}.bias"])


class ResizeBlockCkpt(nn.Module):
    def __init__(self, prefix, ch_in, ch_out, out_hw):
        super().__init__()
        self.out_hw = out_hw
        self.up_conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=True)
        self.block_conv0 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=True)
        self.block_conv1 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=True)
        copy_conv(self.up_conv, f"{prefix}.up_conv")
        copy_conv(self.block_conv0, f"{prefix}.block_conv0")
        copy_conv(self.block_conv1, f"{prefix}.block_conv1")

    def forward(self, x):
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        x = F.relu(self.up_conv(x))
        out = F.relu(self.block_conv0(x))
        out = self.block_conv1(out)
        return F.relu(out + x)


class Dec2ResizeUpConvChunkCkpt(nn.Module):
    def __init__(self, chunk, include_bias):
        super().__init__()
        self.conv = nn.Conv2d(80, 32, kernel_size=3, padding=1, bias=include_bias)
        copy_conv(self.conv, "backbone.dec2.up_conv", slice(chunk * 80, (chunk + 1) * 80))

    def forward(self, x):
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        return self.conv(x)


class Dec2BlockConv0Ckpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        copy_conv(self.conv, "backbone.dec2.block_conv0")

    def forward(self, x):
        return F.relu(self.conv(x))


class Dec2BlockConv1Ckpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        copy_conv(self.conv, "backbone.dec2.block_conv1")

    def forward(self, x):
        return self.conv(x)


class HeadConvCkpt(nn.Module):
    def __init__(self, prefix, ch_in, ch_out, relu=False, sigmoid=False):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=True)
        self.relu = relu
        self.sigmoid = sigmoid
        copy_conv(self.conv, f"{prefix}.conv")

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x


class HeadConvPaddedOutCkpt(nn.Module):
    def __init__(self, prefix, ch_in, ch_out, padded_out, relu=False, sigmoid=False):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, padded_out, kernel_size=3, padding=1, bias=True)
        self.relu = relu
        self.sigmoid = sigmoid
        net = ckpt_net()
        with torch.no_grad():
            weight = net[f"{prefix}.conv.weight"]
            bias = net[f"{prefix}.conv.bias"]
            for idx in range(padded_out):
                src = idx if idx < ch_out else 0
                self.conv.weight[idx].copy_(weight[src])
                self.conv.bias[idx].copy_(bias[src])

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x


class HeadConvPaddedOutVarCkpt(nn.Module):
    def __init__(self, prefix, ch_in, ch_out, padded_out, relu=False, sigmoid=False):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, padded_out, kernel_size=3, padding=1, bias=True)
        self.relu = relu
        self.sigmoid = sigmoid
        net = ckpt_net()
        with torch.no_grad():
            weight = net[f"{prefix}.conv.weight"]
            bias = net[f"{prefix}.conv.bias"]
            self.conv.weight[:ch_out].copy_(weight)
            self.conv.bias[:ch_out].copy_(bias)
            for idx in range(ch_out, padded_out):
                factor = 1.0 + 0.01 * idx
                self.conv.weight[idx].copy_(weight[0] * factor)
                self.conv.bias[idx].copy_(bias[0] * factor)

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x
