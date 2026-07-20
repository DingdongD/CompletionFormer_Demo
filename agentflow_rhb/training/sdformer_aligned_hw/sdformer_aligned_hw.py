"""Training-ready SDFormer HW128 aligned model.

This module keeps the upstream SDFormer encoder/decoder topology but replaces
the unsupported window-attention and FFN cores with RHB-friendly Conv3x3 mixers.

It is a retrain-required architecture.  It is not a strict pretrained SDFormer
rewrite.  The deployment contract is:

- Conv/Conv1x1/Conv3x3 run on RHB.
- LayerNorm, residual add, concat/crop, PixelShuffle/Unshuffle and activations
  are Host glue.
- Refinement/full-res 72->72 Conv3x3 mixers are exported as three exact
  output-channel chunks of 72->24 and concatenated on Host.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_3d(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, _n, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str = "WithBias"):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class HostActivation(nn.Module):
    """Activation marker.

    The model uses regular PyTorch activations for training.  The deployment
    exporter treats this module as Host glue and does not include it in RHB
    subgraphs unless a dedicated activation package is later board-proven.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x, inplace=False)


class ConvHostRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=padding, bias=bias)
        self.act = HostActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class RHBConvMixer(nn.Module):
    """Shape-preserving Conv3x3 mixer used for attention and FFN replacement."""

    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HWAlignedBlock(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str = "WithBias", bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attn_mixer = RHBConvMixer(dim, bias=bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.ffn_mixer = RHBConvMixer(dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_mixer(self.norm1(x))
        x = x + self.ffn_mixer(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.conv = nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelUnshuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.conv = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


@dataclass
class SDFormerHW128Config:
    inp_channels: int = 4
    out_channels: int = 1
    dim: int = 24
    num_blocks: List[int] = None
    num_refinement_blocks: int = 2
    bias: bool = False
    layer_norm_type: str = "WithBias"
    lr: float = 3e-4

    def __post_init__(self) -> None:
        if self.num_blocks is None:
            self.num_blocks = [2, 4, 6, 8]


def make_stage(blocks: int, dim: int, layer_norm_type: str, bias: bool) -> nn.Sequential:
    return nn.Sequential(*[HWAlignedBlock(dim, layer_norm_type, bias=bias) for _ in range(blocks)])


class SDFormerAlignedHW128(nn.Module):
    def __init__(self, cfg: SDFormerHW128Config | None = None):
        super().__init__()
        self.cfg = cfg or SDFormerHW128Config()
        dim = self.cfg.dim
        bias = self.cfg.bias
        ln_type = self.cfg.layer_norm_type

        self.conv1_rgb = ConvHostRelu(3, 18, kernel=3, padding=1, bias=False)
        self.conv1_dep = ConvHostRelu(1, 6, kernel=3, padding=1, bias=False)

        self.encoder_level1 = make_stage(self.cfg.num_blocks[0], dim, ln_type, bias)
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = make_stage(self.cfg.num_blocks[1], dim * 2, ln_type, bias)
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = make_stage(self.cfg.num_blocks[2], dim * 4, ln_type, bias)
        self.down3_4 = Downsample(dim * 4)

        self.latent = make_stage(self.cfg.num_blocks[3], dim * 8, ln_type, bias)

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = make_stage(self.cfg.num_blocks[2], dim * 4, ln_type, bias)

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = make_stage(self.cfg.num_blocks[1], dim * 2, ln_type, bias)

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = make_stage(self.cfg.num_blocks[0], dim * 2, ln_type, bias)

        self.refinement = make_stage(self.cfg.num_refinement_blocks, dim * 3, ln_type, bias)
        self.output = nn.Conv2d(dim * 3, self.cfg.out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        params = [param for param in self.parameters() if param.requires_grad]
        self.param_groups = [{"params": params, "lr": self.cfg.lr}]

    @staticmethod
    def _before_down(x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = torch.cat((x, x[:, :, -1:, :]), dim=2)
        if w % 2 != 0:
            x = torch.cat((x, x[:, :, :, -1:]), dim=3)
        return x

    @staticmethod
    def _concat(fd: torch.Tensor, fe: torch.Tensor, dim: int = 1) -> torch.Tensor:
        _, _, hd, wd = fd.shape
        _, _, he, we = fe.shape
        if hd > he:
            fd = fd[:, :, :he, :]
        if wd > we:
            fd = fd[:, :, :, :we]
        return torch.cat((fd, fe), dim=dim)

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rgb = sample["rgb"]
        dep = sample["dep"]

        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)

        inp_enc_level1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(self._before_down(out_enc_level1))
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(self._before_down(out_enc_level2))
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(self._before_down(out_enc_level3))
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = self._concat(inp_dec_level3, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self._concat(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self._concat(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self._concat(out_dec_level1, inp_enc_level1)
        r1 = self.refinement(out_dec_level1)

        pred = torch.clamp(self.output(r1), min=0)
        return {"pred": pred}

    def rhb_export_hints(self) -> Dict[str, object]:
        """Return the current deployment-specific split contracts."""
        conv_mixers = []
        for name, module in self.named_modules():
            if isinstance(module, RHBConvMixer):
                ch = module.conv.in_channels
                if ch == 72:
                    conv_mixers.append({"name": name, "contract": "output_split", "chunks": [24, 24, 24]})
                else:
                    conv_mixers.append({"name": name, "contract": "single", "chunks": [ch]})
        return {
            "input_shape": [1, 3, 128, 128],
            "depth_shape": [1, 1, 128, 128],
            "attention_and_ffn_mixers": conv_mixers,
            "final_head": {"contract": "input_channel_split", "note": "reuse SDFormer final-head split rule"},
            "host_glue": ["LayerNorm2d", "HostActivation", "PixelShuffle", "PixelUnshuffle", "concat", "residual", "clamp"],
        }


def count_parameters(model: nn.Module) -> int:
    return sum(math.prod(p.shape) for p in model.parameters())
