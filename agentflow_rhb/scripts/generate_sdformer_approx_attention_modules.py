#!/usr/bin/env python3
"""Generate retrainable SDFormer attention replacement probes.

The strict SDFormer window-attention core is not board-accepted today.  These
modules are latency-first, retrain-required candidates that preserve the tensor
contract:

    [B, C, H, W] -> [B, C, H, W]

but replace qkv_dwconv + window normalize/matmul/softmax + project_out with a
plain Conv/ReLU mixer that is intended to be RHB-friendly.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/root/demo/models/sdformer_test/approx_generated")


CONV_MIXER_TEMPLATE = """import torch
import torch.nn as nn

from models.sdformer_test._common import init_module

op_version = 18
ifmap_sz = [({channels}, {spatial}, {spatial})]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d({channels}, {channels}, kernel_size=3, stride=1, padding=1, bias=False)
        init_module(self)

    def forward(self, x):
        y = self.conv(x)
{activation}
        return y
"""


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in ROOT.glob("*.py"):
        if path.name != "__init__.py":
            path.unlink()
    (ROOT / "__init__.py").write_text("", encoding="utf-8")

    # Encoder/decoder/refinement SDFormer HW128 channel contracts.
    cases = [
        (24, 128),
        (48, 64),
        (96, 32),
        (192, 16),
        (72, 128),
    ]
    for channels, spatial in cases:
        name = f"attn_approx_conv3x3_c{channels}_{spatial}"
        (ROOT / f"{name}.py").write_text(
            CONV_MIXER_TEMPLATE.format(
                channels=channels,
                spatial=spatial,
                activation="",
            ),
            encoding="utf-8",
        )

    # Exact fallback for the only failing full-res/high-channel mixer:
    # 72 -> 72 at 128x128.  Output channels are independent, so three
    # 72 -> 24 Conv3x3 launches plus Host concat are semantically exact when
    # the real checkpoint weights are sliced by output channel.
    for idx in range(3):
        name = f"attn_approx_conv3x3_c72_128_oc{idx}_out24"
        (ROOT / f"{name}.py").write_text(
            CONV_MIXER_TEMPLATE.format(
                channels=72,
                spatial=128,
                activation="",
            ).replace("nn.Conv2d(72, 72,", "nn.Conv2d(72, 24,"),
            encoding="utf-8",
        )

    print(f"Generated {len(list(ROOT.glob('*.py'))) - 1} SDFormer approximate attention modules in {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
