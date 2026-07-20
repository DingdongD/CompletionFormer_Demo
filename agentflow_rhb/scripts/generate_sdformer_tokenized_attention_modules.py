#!/usr/bin/env python3
"""Generate exact tokenized SDFormer window-attention probe modules.

These modules model the RHB candidate core after Host-side window packing:

    q/k/v: [B*Nw, Cg, P]

where Cg = D / 3 and P = dh * dw.  This is mathematically equivalent to the
original SDFormer window channel-attention core, but avoids putting window
pack/unpack into the RHB subgraph.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/root/demo/models/sdformer_test/tokenized_generated")


QK_TEMPLATE = """import torch
import torch.nn as nn

op_version = 18
input_layouts = ["WC", "WC"]
ifmap_sz = [({cg}, {p}), ({cg}, {p})]


class Model(nn.Module):
    def forward(self, q, k):
        return q @ k.transpose(-2, -1)
"""


SOFTMAX_TEMPLATE = """import torch
import torch.nn as nn
import torch.nn.functional as F

op_version = 18
input_layouts = ["WC"]
ifmap_sz = [({cg}, {cg})]


class Model(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=-1)
"""


AV_TEMPLATE = """import torch
import torch.nn as nn

op_version = 18
input_layouts = ["WC", "WC"]
ifmap_sz = [({cg}, {cg}), ({cg}, {p})]


class Model(nn.Module):
    def forward(self, attn, v):
        return attn @ v
"""


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "__init__.py").write_text("", encoding="utf-8")

    # Unique (Cg, P) pairs for SDFormer HW128:
    # level1: D=24 -> Cg=8,    P=256/64/16
    # level2: D=48 -> Cg=16,   P=256/64/16
    # level3: D=96 -> Cg=32,   P=64/16
    # level4: D=192 -> Cg=64,  P=16
    # refinement: D=72 -> Cg=24, P=256/64/16
    pairs = [
        (8, 16), (8, 64), (8, 256),
        (16, 16), (16, 64), (16, 256),
        (24, 16), (24, 64), (24, 256),
        (32, 16), (32, 64),
        (64, 16),
    ]

    for cg, p in pairs:
        (ROOT / f"tokenized_window_qk_cg{cg}_p{p}.py").write_text(
            QK_TEMPLATE.format(cg=cg, p=p), encoding="utf-8"
        )
        (ROOT / f"tokenized_window_softmax_cg{cg}_p{p}.py").write_text(
            SOFTMAX_TEMPLATE.format(cg=cg, p=p), encoding="utf-8"
        )
        (ROOT / f"tokenized_window_av_cg{cg}_p{p}.py").write_text(
            AV_TEMPLATE.format(cg=cg, p=p), encoding="utf-8"
        )

    print(f"Generated {len(pairs) * 3} tokenized SDFormer attention modules in {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
