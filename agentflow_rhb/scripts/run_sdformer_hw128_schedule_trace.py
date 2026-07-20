#!/usr/bin/env python3
"""Trace the SDFormer HW128 Host/RHB schedule.

This script is intentionally checkpoint-agnostic. It instantiates the upstream
SDFormer model with 128x128-compatible window sizes, runs a full forward pass,
and records the module-level Host/RHB allocation implied by the current
AgentFlow SDFormer rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn as nn


def make_args() -> SimpleNamespace:
    # Each window size must divide the spatial resolution at that stage:
    # stage1/refinement 128x128, stage2 64x64, stage3 32x32, stage4 16x16.
    return SimpleNamespace(
        inp_channels=4,
        out_channels=1,
        dim=24,
        heads=[1, 2, 4, 8],
        num_blocks=[2, 4, 6, 8],
        num_refinement_blocks=2,
        ffn_expansion_factor=2.88,
        bias=False,
        LayerNorm_type="WithBias",
        lr=3e-4,
        window_sizes1=[[16, 16], [8, 8], [4, 4]],
        window_sizes2=[[16, 16], [8, 8], [4, 4]],
        window_sizes3=[[8, 8], [4, 4], [4, 4]],
        window_sizes4=[[4, 4], [4, 4], [4, 4]],
    )


def classify_module(name: str, module: nn.Module) -> str:
    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels == module.out_channels and module.groups > 1:
            return "HOST_STRICT_FALLBACK_DEPTHWISE"
        if name.endswith("output"):
            return "RHB_EXACT_IC_SPLIT"
        return "RHB_CANDIDATE_BOARD_PROVEN_BY_PATTERN"
    cls = module.__class__.__name__
    if cls == "LayerNorm":
        return "HOST_LAYER_NORM"
    if cls in {"Attention", "FeedForward", "TransformerBlock"}:
        return "COMPOSITE_HOST_RHB_SPLIT"
    if cls in {"Downsample", "Upsample"}:
        return "COMPOSITE_RHB_CONV_HOST_PIXEL_SHUFFLE"
    return "HOST_OR_CONTAINER"


def shape_of(value) -> List[int] | str:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        return [list(v.shape) for v in value]
    return type(value).__name__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/root/demo/SDformer-for-Depth-Completion")
    parser.add_argument("--out", default="/root/demo/artifacts/output_sdformer_probe_20260717/sdformer_hw128_schedule_trace.json")
    parser.add_argument("--seed", type=int, default=240717)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    sys.path.insert(0, str(source_root / "src"))
    from model.sdformermodel import SDFORMERModel  # pylint: disable=import-error

    torch.manual_seed(args.seed)
    model = SDFORMERModel(make_args()).eval()

    events: List[Dict[str, object]] = []

    def hook(name: str, module: nn.Module):
        def _inner(_module, inputs, output):
            decision = classify_module(name, module)
            if decision != "HOST_OR_CONTAINER":
                events.append(
                    {
                        "name": name,
                        "class": module.__class__.__name__,
                        "decision": decision,
                        "input_shape": shape_of(inputs),
                        "output_shape": shape_of(output),
                        "groups": getattr(module, "groups", None),
                        "kernel_size": list(getattr(module, "kernel_size", [])),
                    }
                )

        return _inner

    handles = [m.register_forward_hook(hook(n, m)) for n, m in model.named_modules() if n]
    with torch.no_grad():
        rgb = torch.randn(1, 3, 128, 128)
        dep = torch.randn(1, 1, 128, 128)
        pred = model({"rgb": rgb, "dep": dep})["pred"]
    for handle in handles:
        handle.remove()

    summary: Dict[str, int] = {}
    for event in events:
        summary[event["decision"]] = summary.get(event["decision"], 0) + 1

    out = {
        "model": "SDFormer HW128 schedule trace",
        "source_root": str(source_root),
        "checkpoint": None,
        "input": {"rgb": [1, 3, 128, 128], "dep": [1, 1, 128, 128]},
        "output": {"pred": list(pred.shape), "min": float(pred.min()), "max": float(pred.max())},
        "summary": summary,
        "events": events,
        "note": "This is a full CPU forward plus allocation trace. Board-proven probe subgraphs are listed in sdformer_hw128_adaptation.md.",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "output_shape": list(pred.shape), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
