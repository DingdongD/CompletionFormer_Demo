import torch
import torch.nn as nn
import torch.nn.functional as F


input_layouts = ["CHW"]
ifmap_sz = [[4, 32, 32]]
op_version = 14
batch_size = 1


class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        return F.relu(self.conv(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_ch = 8
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.stem0 = ConvReLU(4, c1, 3, 1)
        self.stem1 = ConvReLU(c1, c1, 3, 1)
        self.down10 = ConvReLU(c1, c2, 3, 2)
        self.down11 = ConvReLU(c2, c2, 3, 1)
        self.down20 = ConvReLU(c2, c3, 3, 2)
        self.down21 = ConvReLU(c3, c3, 3, 1)
        self.down22 = ConvReLU(c3, c3, 3, 1)

        self.up1_resize = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up10 = ConvReLU(c3 + c2, c2, 3, 1)
        self.up11 = ConvReLU(c2, c2, 3, 1)
        self.up2_resize = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up20 = ConvReLU(c2 + c1, c1, 3, 1)
        self.up21 = ConvReLU(c1, c1, 3, 1)

        self.depth0 = ConvReLU(c1, c1, 3, 1)
        self.depth1 = nn.Conv2d(c1, 1, kernel_size=3, padding=1, bias=True)
        self.aff0 = ConvReLU(c1, c1, 3, 1)
        self.aff1 = nn.Conv2d(c1, 8, kernel_size=3, padding=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for op in self.modules():
            if isinstance(op, nn.Conv2d):
                nn.init.xavier_uniform_(op.weight, gain=0.1)
                if op.bias is not None:
                    nn.init.uniform_(op.bias, -0.1, 0.1)

    def forward(self, x):
        s1 = self.stem1(self.stem0(x))
        s2 = self.down11(self.down10(s1))
        b = self.down22(self.down21(self.down20(s2)))

        u1 = self.up1_resize(b)
        u1 = torch.cat([u1, s2], dim=1)
        u1 = self.up11(self.up10(u1))

        u2 = self.up2_resize(u1)
        u2 = torch.cat([u2, s1], dim=1)
        feat = self.up21(self.up20(u2))

        raw_depth = self.depth1(self.depth0(feat))
        raw_guidance = self.aff1(self.aff0(feat))
        return raw_depth, raw_guidance
