#!/usr/bin/env python3
"""SDFormer HW128 Host/RHB orchestrated reference runner.

This runner executes the complete SDFormer forward path on Host PyTorch while
emitting the exact Host/RHB allocation contract used by AgentFlow. It is the
checkpoint/calibration handoff point for the later board runner:

- Conv/Conv1x1/selected final-head split chunks are RHB candidates.
- LayerNorm, depthwise group Conv, window attention core, PixelShuffle,
  PixelUnshuffle, concat/crop, residual add, GELU/gate multiply and clamp are
  Host glue unless a later board proof promotes them.

No local SDFormer checkpoint is present at the time this script was added; if a
checkpoint is supplied, it is loaded before inference and the strict load report
is written into the output JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn


def make_hw128_args() -> SimpleNamespace:
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


def load_checkpoint(model: nn.Module, ckpt_path: str | None) -> Dict[str, Any]:
    if not ckpt_path:
        return {"path": None, "loaded": False, "reason": "no checkpoint provided"}
    path = Path(ckpt_path)
    if not path.exists():
        return {"path": str(path), "loaded": False, "reason": "checkpoint file not found"}
    obj = torch.load(path, map_location="cpu")
    state = obj
    for key in ("state_dict", "model", "net", "model_state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            state = obj[key]
            break
    if not isinstance(state, dict):
        return {"path": str(path), "loaded": False, "reason": "checkpoint did not contain a state dict"}
    cleaned = {}
    for key, value in state.items():
        new_key = str(key)
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return {
        "path": str(path),
        "loaded": True,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "strict_match": not missing and not unexpected,
    }


def classify(name: str, module: nn.Module) -> str:
    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels == module.out_channels and module.groups > 1:
            return "HOST_DEPTHWISE_GROUP_CONV"
        if name.endswith("output"):
            return "RHB_EXACT_FINAL_HEAD_IC_SPLIT"
        if name.endswith("project_in"):
            return "RHB_EXACT_FFN_PROJECT_IN_OC_SPLIT72"
        return "RHB_CONV_SINGLE_OUTPUT"
    cls = module.__class__.__name__
    if cls == "LayerNorm":
        return "HOST_LAYER_NORM"
    if cls == "Attention":
        return "HOST_RHB_SPLIT_ATTENTION"
    if cls == "FeedForward":
        return "HOST_RHB_SPLIT_FFN"
    if cls in {"Downsample", "Upsample"}:
        return "HOST_RHB_SPLIT_SHUFFLE_BLOCK"
    if cls == "TransformerBlock":
        return "HOST_RHB_SPLIT_TRANSFORMER_BLOCK"
    return "HOST_CONTAINER_OR_GLUE"


def tensor_shape(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        return [tensor_shape(v) for v in x]
    if isinstance(x, dict):
        return {k: tensor_shape(v) for k, v in x.items()}
    return type(x).__name__


def load_sample(path: str | None) -> Dict[str, torch.Tensor]:
    if not path:
        torch.manual_seed(240720)
        return {"rgb": torch.randn(1, 3, 128, 128), "dep": torch.rand(1, 1, 128, 128)}
    data = np.load(path)
    rgb_key = "rgb" if "rgb" in data else "images" if "images" in data else None
    dep_key = "dep" if "dep" in data else "sparse_dep" if "sparse_dep" in data else "depth" if "depth" in data else None
    if rgb_key is None or dep_key is None:
        raise KeyError(f"Could not find rgb/dep arrays in {path}; keys={list(data.keys())}")
    rgb = torch.from_numpy(np.asarray(data[rgb_key])[:1]).float()
    dep = torch.from_numpy(np.asarray(data[dep_key])[:1]).float()
    if rgb.ndim == 4 and rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)
    if dep.ndim == 3:
        dep = dep[:, None]
    if dep.ndim == 4 and dep.shape[-1] == 1:
        dep = dep.permute(0, 3, 1, 2)
    return {"rgb": rgb[:, :3, :128, :128], "dep": dep[:, :1, :128, :128]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/root/demo/SDformer-for-Depth-Completion")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--input-npz", default="")
    parser.add_argument("--out-dir", default="/root/demo/artifacts/output_sdformer_hw128_orchestrated_20260720")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.source_root) / "src"))
    from model.sdformermodel import SDFORMERModel  # pylint: disable=import-error

    model = SDFORMERModel(make_hw128_args()).eval()
    ckpt_report = load_checkpoint(model, args.checkpoint or None)

    events: List[Dict[str, Any]] = []

    def hook(name: str, module: nn.Module):
        def _inner(_module, inputs, output):
            decision = classify(name, module)
            if decision == "HOST_CONTAINER_OR_GLUE":
                return
            events.append(
                {
                    "name": name,
                    "class": module.__class__.__name__,
                    "decision": decision,
                    "input_shape": tensor_shape(inputs),
                    "output_shape": tensor_shape(output),
                    "groups": getattr(module, "groups", None),
                    "kernel_size": list(getattr(module, "kernel_size", [])),
                }
            )
        return _inner

    handles = [m.register_forward_hook(hook(n, m)) for n, m in model.named_modules() if n]
    sample = load_sample(args.input_npz or None)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(sample)["pred"]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    for handle in handles:
        handle.remove()

    summary: Dict[str, int] = {}
    for event in events:
        summary[event["decision"]] = summary.get(event["decision"], 0) + 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "sdformer_hw128_orchestrated_outputs.npz",
        rgb=sample["rgb"].numpy(),
        dep=sample["dep"].numpy(),
        pred=pred.numpy(),
    )
    report = {
        "model": "SDFormer HW128 Host/RHB orchestrated reference",
        "source_root": str(Path(args.source_root)),
        "checkpoint": ckpt_report,
        "input_npz": args.input_npz or None,
        "output_shape": list(pred.shape),
        "host_pytorch_latency_ms": elapsed_ms,
        "allocation_summary": summary,
        "events": events,
        "board_status": "not_run_by_this_script",
        "note": "This is the exact reference/allocation runner. Board execution uses generated probe modules and packers once checkpoint-owned weights are exported.",
    }
    (out_dir / "sdformer_hw128_orchestrated_schedule.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("output_shape", "host_pytorch_latency_ms", "allocation_summary")}, indent=2))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
