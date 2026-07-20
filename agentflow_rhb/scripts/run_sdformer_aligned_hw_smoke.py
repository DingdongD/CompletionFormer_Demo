#!/usr/bin/env python3
"""Smoke-test the training-ready SDFormer HW128 aligned model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PORT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORT))

from agentflow_rhb.training.sdformer_aligned_hw.sdformer_aligned_hw import (  # noqa: E402
    SDFormerAlignedHW128,
    SDFormerHW128Config,
    count_parameters,
)


def load_sample(path: str | None) -> dict[str, torch.Tensor]:
    if not path:
        torch.manual_seed(240720)
        return {
            "rgb": torch.randn(1, 3, 128, 128),
            "dep": torch.rand(1, 1, 128, 128),
        }
    data = np.load(path)
    rgb_key = "rgb" if "rgb" in data else "images" if "images" in data else None
    dep_key = "dep" if "dep" in data else "sparse_dep" if "sparse_dep" in data else "depth" if "depth" in data else None
    if rgb_key is None or dep_key is None:
        raise KeyError(f"Could not find rgb/depth arrays in {path}; keys={list(data.keys())}")
    rgb = torch.from_numpy(np.asarray(data[rgb_key])[:1]).float()
    dep = torch.from_numpy(np.asarray(data[dep_key])[:1]).float()
    if rgb.ndim == 4 and rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)
    if dep.ndim == 3:
        dep = dep[:, None]
    if dep.ndim == 4 and dep.shape[-1] == 1:
        dep = dep.permute(0, 3, 1, 2)
    return {
        "rgb": rgb[:, :3, :128, :128].contiguous(),
        "dep": dep[:, :1, :128, :128].contiguous(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out-dir", default="/root/demo/artifacts/output_sdformer_aligned_hw_smoke_20260720")
    args = parser.parse_args()

    model = SDFormerAlignedHW128(SDFormerHW128Config()).eval()
    ckpt_report = {"path": None, "loaded": False}
    if args.checkpoint:
        obj = torch.load(args.checkpoint, map_location="cpu")
        state = obj
        for key in ("state_dict", "model", "net", "model_state_dict"):
            if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        ckpt_report = {
            "path": args.checkpoint,
            "loaded": True,
            "missing": list(missing),
            "unexpected": list(unexpected),
            "strict_match": not missing and not unexpected,
        }

    sample = load_sample(args.input_npz or None)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(sample)["pred"]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "sdformer_aligned_hw_smoke_outputs.npz",
        rgb=sample["rgb"].numpy(),
        dep=sample["dep"].numpy(),
        pred=pred.numpy(),
    )
    report = {
        "model": "SDFormerAlignedHW128",
        "checkpoint": ckpt_report,
        "input_npz": args.input_npz or None,
        "output_shape": list(pred.shape),
        "host_latency_ms": latency_ms,
        "parameters": count_parameters(model),
        "rhb_export_hints": model.rhb_export_hints(),
    }
    (out_dir / "sdformer_aligned_hw_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("output_shape", "host_latency_ms", "parameters")}, indent=2))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
