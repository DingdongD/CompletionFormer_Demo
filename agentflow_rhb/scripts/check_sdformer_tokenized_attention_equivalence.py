#!/usr/bin/env python3
"""Check exact SDFormer window-attention tokenization against the source form."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def pack_windows(x: torch.Tensor, dh: int, dw: int) -> torch.Tensor:
    b, c, h, w = x.shape
    nh, nw = h // dh, w // dw
    return (
        x.reshape(b, c, nh, dh, nw, dw)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .reshape(b, nh * nw, c, dh * dw)
    )


def unpack_windows(x: torch.Tensor, h: int, w: int, dh: int, dw: int) -> torch.Tensor:
    b, nwin, c, p = x.shape
    nh, nw = h // dh, w // dw
    assert nwin == nh * nw
    assert p == dh * dw
    return (
        x.reshape(b, nh, nw, c, dh, dw)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .reshape(b, c, h, w)
    )


def source_branch(x: torch.Tensor, dh: int, dw: int) -> torch.Tensor:
    b, d, h, w = x.shape
    q, k, v = x.chunk(3, dim=1)
    q = pack_windows(q, dh, dw)
    k = pack_windows(k, dh, dw)
    v = pack_windows(v, dh, dw)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    out = F.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
    return unpack_windows(out, h, w, dh, dw)


def tokenized_branch(x: torch.Tensor, dh: int, dw: int) -> torch.Tensor:
    b, d, h, w = x.shape
    q, k, v = x.chunk(3, dim=1)
    q = pack_windows(q, dh, dw)
    k = pack_windows(k, dh, dw)
    v = pack_windows(v, dh, dw)
    b0, nwin, cg, p = q.shape

    # Exact tokenized form: each window is a batch item.
    qt = F.normalize(q.reshape(b0 * nwin, cg, p), dim=-1)
    kt = F.normalize(k.reshape(b0 * nwin, cg, p), dim=-1)
    vt = v.reshape(b0 * nwin, cg, p)
    out = (F.softmax(qt @ kt.transpose(-2, -1), dim=-1) @ vt).reshape(b0, nwin, cg, p)
    return unpack_windows(out, h, w, dh, dw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/demo/artifacts/output_sdformer_tokenized_attention_matrix_20260720/equivalence.json")
    args = parser.parse_args()

    torch.manual_seed(240720)
    cases = [
        {"dim": 24, "h": 128, "w": 128, "windows": [(16, 16), (8, 8), (4, 4)]},
        {"dim": 48, "h": 64, "w": 64, "windows": [(16, 16), (8, 8), (4, 4)]},
        {"dim": 96, "h": 32, "w": 32, "windows": [(8, 8), (4, 4), (4, 4)]},
        {"dim": 192, "h": 16, "w": 16, "windows": [(4, 4), (4, 4), (4, 4)]},
        {"dim": 72, "h": 128, "w": 128, "windows": [(16, 16), (8, 8), (4, 4)]},
    ]
    results = []
    for case in cases:
        for dh, dw in case["windows"]:
            x = torch.randn(1, case["dim"], case["h"], case["w"])
            a = source_branch(x, dh, dw)
            b = tokenized_branch(x, dh, dw)
            diff = (a - b).abs()
            results.append({
                "dim": case["dim"],
                "h": case["h"],
                "w": case["w"],
                "window": [dh, dw],
                "cg": case["dim"] // 3,
                "p": dh * dw,
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
