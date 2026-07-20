#!/usr/bin/env python3
"""Generate SDFormer HW128 RHB candidate probe modules.

The generated modules are shape/probe modules. They intentionally use
deterministic random weights; checkpoint-owned weights are injected by the
end-to-end exporter/runner once a SDFormer checkpoint exists.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/root/demo/models/sdformer_test/generated")


HEADER = """import torch
import torch.nn as nn

from models.sdformer_test._common import init_module

op_version = 18
ifmap_sz = [({cin}, {h}, {w})]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d({cin}, {cout}, kernel_size={kernel}, stride=1, padding={padding}, bias={bias})
        init_module(self)
{post_init}

    def forward(self, x):
        y = self.conv(x)
{post_forward}
        return y
"""


INIT_ZERO_TAIL = """        with torch.no_grad():
            self.conv.weight[{real_out}:].zero_()
            if self.conv.bias is not None:
                self.conv.bias[{real_out}:].zero_()
"""


def write_module(name: str, cin: int, cout: int, h: int, w: int, kernel: int, padding: int, *,
                 bias: bool = False, relu: bool = False, real_out: int | None = None) -> None:
    post_init = ""
    if real_out is not None and real_out < cout:
        post_init = INIT_ZERO_TAIL.format(real_out=real_out).rstrip()
    post_forward = "        y = torch.relu(y)\n" if relu else ""
    text = HEADER.format(
        cin=cin,
        cout=cout,
        h=h,
        w=w,
        kernel=kernel,
        padding=padding,
        bias=str(bias),
        post_init=post_init,
        post_forward=post_forward.rstrip(),
    )
    (ROOT / f"{name}.py").write_text(text, encoding="utf-8")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "__init__.py").write_text("", encoding="utf-8")

    # Stem.
    write_module("stem_rgb_conv_relu_3_18_128", 3, 18, 128, 128, 3, 1, relu=True)
    write_module("stem_dep_conv_relu_1_6_128", 1, 6, 128, 128, 3, 1, relu=True)

    # Downsample pre-PixelUnshuffle Conv.
    write_module("down1_pre_pixelunshuffle_conv_24_12_128", 24, 12, 128, 128, 3, 1)
    write_module("down2_pre_pixelunshuffle_conv_48_24_64", 48, 24, 64, 64, 3, 1)
    write_module("down3_pre_pixelunshuffle_conv_96_48_32", 96, 48, 32, 32, 3, 1)

    # Upsample pre-PixelShuffle Conv.
    write_module("up4_pre_pixelshuffle_conv_192_384_16", 192, 384, 16, 16, 3, 1)
    write_module("up3_pre_pixelshuffle_conv_96_192_32", 96, 192, 32, 32, 3, 1)
    write_module("up2_pre_pixelshuffle_conv_48_96_64", 48, 96, 64, 64, 3, 1)

    # Decoder reduce-channel pointwise Conv after Host concat.
    write_module("reduce_level3_conv1x1_192_96_32", 192, 96, 32, 32, 1, 0)
    write_module("reduce_level2_conv1x1_96_48_64", 96, 48, 64, 64, 1, 0)

    # Transformer attention qkv/project pointwise Conv candidates.
    for channels, spatial in [(24, 128), (48, 64), (96, 32), (192, 16), (72, 128)]:
        write_module(f"attn_qkv_conv1x1_{channels}_{channels * 3}_{spatial}", channels, channels * 3, spatial, spatial, 1, 0)
        write_module(f"attn_project_conv1x1_{channels}_{channels}_{spatial}", channels, channels, spatial, spatial, 1, 0)

    # FFN project_in is exact output-channel split into 72-wide RHB chunks.
    # Host keeps only the real channels from the last padded chunk.
    for channels, spatial in [(24, 128), (48, 64), (96, 32), (192, 16), (72, 128)]:
        hidden = int(channels * 2.88)
        total = hidden * 2
        chunks = (total + 71) // 72
        for idx in range(chunks):
            real = min(72, total - idx * 72)
            write_module(
                f"ffn_project_in_oc{idx}_conv1x1_{channels}_72_real{real}_{spatial}",
                channels,
                72,
                spatial,
                spatial,
                1,
                0,
                real_out=real,
            )
        write_module(f"ffn_project_out_conv1x1_{hidden}_{channels}_{spatial}", hidden, channels, spatial, spatial, 1, 0)

    # Final prediction head: exact input-channel split, Host sums output channel0.
    for idx, real_in in enumerate([16, 16, 16, 16, 8]):
        write_module(f"final_head_ic{idx}_conv3x3_{real_in}_8_128", real_in, 8, 128, 128, 3, 1, real_out=1)

    print(f"Generated {len(list(ROOT.glob('*.py'))) - 1} SDFormer probe modules in {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
