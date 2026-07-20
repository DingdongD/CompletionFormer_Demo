#!/usr/bin/env python3
"""Export the SDFormer HW128 hardware-aligned deployment plan.

The SDFormer aligned model is a retrain-required variant.  This script does not
claim pretrained accuracy.  It emits the end-to-end Host/RHB schedule, the RHB
launch contract, and the board evidence required before a trained checkpoint is
accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn


PORTABLE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORTABLE_ROOT))

from agentflow_rhb.training.sdformer_aligned_hw.sdformer_aligned_hw import (  # noqa: E402
    ConvHostRelu,
    Downsample,
    HostActivation,
    LayerNorm2d,
    RHBConvMixer,
    SDFormerAlignedHW128,
    SDFormerHW128Config,
    Upsample,
    count_parameters,
)


FINAL_HEAD_IC_CHUNKS = [16, 16, 16, 16, 8]
REFINEMENT_OC_CHUNKS = [24, 24, 24]


def load_sample(path: str | None) -> Dict[str, torch.Tensor]:
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
        raise KeyError(f"Could not find rgb/dep arrays in {path}; keys={list(data.keys())}")
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


def load_checkpoint(model: nn.Module, checkpoint: str) -> Dict[str, Any]:
    if not checkpoint:
        return {"path": None, "loaded": False, "reason": "no checkpoint provided"}
    path = Path(checkpoint)
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
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return {
        "path": str(path),
        "loaded": True,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "strict_match": not missing and not unexpected,
    }


def shape_of(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shape_of(v) for v in value]
    if isinstance(value, dict):
        return {k: shape_of(v) for k, v in value.items()}
    return type(value).__name__


def conv_contract(name: str, module: nn.Conv2d) -> Dict[str, Any]:
    in_ch = module.in_channels
    out_ch = module.out_channels
    kernel = list(module.kernel_size)
    groups = module.groups
    if name == "output":
        return {
            "placement": "RHB",
            "contract": "input_channel_split",
            "chunks": FINAL_HEAD_IC_CHUNKS,
            "launches": len(FINAL_HEAD_IC_CHUNKS),
            "host_glue": "sum partial outputs, add final bias once if present, clamp on Host",
            "board_evidence": "sdformer_test.final_head_chunk16_padded8_128 All same=True",
        }
    if groups != 1:
        return {
            "placement": "HOST",
            "contract": "fallback",
            "chunks": [],
            "launches": 0,
            "reason": "group/depthwise Conv is not board-accepted for SDFormer",
        }
    return {
        "placement": "RHB",
        "contract": "single_conv",
        "chunks": [out_ch],
        "launches": 1,
        "board_evidence": "shape covered by SDFormer HW128 Conv matrix or exact probe",
        "kernel": kernel,
        "in_channels": in_ch,
        "out_channels": out_ch,
    }


def mixer_contract(name: str, module: RHBConvMixer) -> Dict[str, Any]:
    channels = module.conv.in_channels
    if channels == 72:
        return {
            "placement": "RHB",
            "contract": "output_channel_split",
            "chunks": REFINEMENT_OC_CHUNKS,
            "launches": len(REFINEMENT_OC_CHUNKS),
            "host_glue": "concatenate three 72->24 Conv3x3 outputs along channel dimension",
            "board_evidence": "c72_128 full package fails; c72_128_oc0/1/2_out24 All same=True",
        }
    return {
        "placement": "RHB",
        "contract": "single_conv3x3_mixer",
        "chunks": [channels],
        "launches": 1,
        "board_evidence": f"Conv3x3 {channels}->{channels} board All same=True for HW128 stage shape",
    }


def capture_schedule(model: SDFormerAlignedHW128, sample: Dict[str, torch.Tensor]) -> tuple[List[Dict[str, Any]], float, torch.Tensor]:
    events: List[Dict[str, Any]] = []
    wrapper_prefixes = {
        name
        for name, module in model.named_modules()
        if name and isinstance(module, (RHBConvMixer, ConvHostRelu, Downsample, Upsample))
    }

    def should_trace(name: str, module: nn.Module) -> bool:
        for prefix in wrapper_prefixes:
            if name.startswith(prefix + "."):
                return False
        if isinstance(module, (RHBConvMixer, ConvHostRelu, Downsample, Upsample, LayerNorm2d, HostActivation)):
            return True
        if isinstance(module, nn.Conv2d):
            return True
        return False

    def hook(name: str, module: nn.Module):
        def _inner(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            entry: Dict[str, Any] = {
                "name": name,
                "class": module.__class__.__name__,
                "input_shape": shape_of(inputs),
                "output_shape": shape_of(output),
            }
            if isinstance(module, RHBConvMixer):
                entry.update(mixer_contract(name, module))
            elif isinstance(module, ConvHostRelu):
                entry.update(
                    {
                        "placement": "RHB+HOST",
                        "contract": "single_conv_then_host_relu",
                        "launches": 1,
                        "board_evidence": "stem Conv+ReLU probe passed; default aligned policy keeps activation as Host glue",
                    }
                )
            elif isinstance(module, Downsample):
                entry.update(
                    {
                        "placement": "RHB+HOST",
                        "contract": "single_conv_then_host_pixel_unshuffle",
                        "launches": 1,
                    }
                )
            elif isinstance(module, Upsample):
                entry.update(
                    {
                        "placement": "RHB+HOST",
                        "contract": "single_conv_then_host_pixel_shuffle",
                        "launches": 1,
                    }
                )
            elif isinstance(module, LayerNorm2d):
                entry.update({"placement": "HOST", "contract": "layer_norm2d", "launches": 0})
            elif isinstance(module, HostActivation):
                entry.update({"placement": "HOST", "contract": "relu", "launches": 0})
            elif isinstance(module, nn.Conv2d):
                entry.update(conv_contract(name, module))
            else:
                entry.update({"placement": "HOST", "contract": "container_or_glue", "launches": 0})
            events.append(entry)

        return _inner

    handles = [module.register_forward_hook(hook(name, module)) for name, module in model.named_modules() if name and should_trace(name, module)]
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(sample)["pred"]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    for handle in handles:
        handle.remove()
    return events, latency_ms, pred


def summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rhb_launches": 0,
        "rhb_events": 0,
        "host_events": 0,
        "contracts": {},
    }
    for event in events:
        placement = event.get("placement", "")
        contract = event.get("contract", "unknown")
        summary["contracts"][contract] = summary["contracts"].get(contract, 0) + 1
        summary["rhb_launches"] += int(event.get("launches", 0))
        if "RHB" in placement:
            summary["rhb_events"] += 1
        if "HOST" in placement:
            summary["host_events"] += 1
    return summary


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# SDFormer HW128 Aligned Deployment Plan",
        "",
        "This is a retrain-required hardware-aligned SDFormer replacement.",
        "It preserves the encoder/decoder tensor topology, but replaces unsupported window attention and FFN cores with RHB-proven Conv3x3 mixers.",
        "",
        "## Status",
        "",
        f"- checkpoint loaded: `{report['checkpoint']['loaded']}`",
        f"- strict checkpoint match: `{report['checkpoint'].get('strict_match', False)}`",
        f"- output shape: `{report['output_shape']}`",
        f"- Host PyTorch reference latency: `{report['host_pytorch_latency_ms']:.3f} ms`",
        f"- planned RHB launches: `{report['schedule_summary']['rhb_launches']}`",
        "- board accuracy status: `pending trained aligned checkpoint + calibration`",
        "- component board status: `accepted for all scheduled Conv3x3 mixer contracts except blacklisted full 72->72, which is split`",
        "",
        "## Contract Summary",
        "",
        "| Contract | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(report["schedule_summary"]["contracts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Board-Safe Replacement",
            "",
            "- strict SDFormer window attention remains Host-only in the rule DB;",
            "- production SDFormer-on-RHB should use `SDFormerAlignedHW128` and retrain;",
            "- `C=72, 128x128` Conv3x3 mixers must be exported as three `72->24` output chunks;",
            "- final head uses exact input-channel split and Host partial-sum glue.",
            "",
            "## Files",
            "",
            "- model: `agentflow_rhb/training/sdformer_aligned_hw/sdformer_aligned_hw.py`",
            "- train: `agentflow_rhb/training/train_sdformer_aligned_hw.py`",
            "- plan json: `sdformer_aligned_hw_deployment_plan.json`",
            "- reference npz: `sdformer_aligned_hw_reference_outputs.npz`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--input-npz", default="")
    parser.add_argument("--out-dir", default="/root/demo/artifacts/output_sdformer_aligned_hw_deployment_plan_20260720")
    args = parser.parse_args()

    model = SDFormerAlignedHW128(SDFormerHW128Config()).eval()
    ckpt_report = load_checkpoint(model, args.checkpoint)
    sample = load_sample(args.input_npz or None)
    events, latency_ms, pred = capture_schedule(model, sample)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "sdformer_aligned_hw_reference_outputs.npz",
        rgb=sample["rgb"].numpy(),
        dep=sample["dep"].numpy(),
        pred=pred.numpy(),
    )
    report: Dict[str, Any] = {
        "model": "SDFormerAlignedHW128",
        "config": asdict(SDFormerHW128Config()),
        "checkpoint": ckpt_report,
        "requires_retraining": True,
        "input_npz": args.input_npz or None,
        "output_shape": list(pred.shape),
        "host_pytorch_latency_ms": latency_ms,
        "parameters": count_parameters(model),
        "rhb_export_hints": model.rhb_export_hints(),
        "schedule_summary": summarize(events),
        "schedule": events,
        "component_board_evidence": {
            "summary_tsv": "/root/demo/artifacts/output_sdformer_approx_attention_matrix_clean_20260720/summary.tsv",
            "accepted": [
                "Conv3x3 24->24 @128x128",
                "Conv3x3 48->48 @64x64",
                "Conv3x3 96->96 @32x32",
                "Conv3x3 192->192 @16x16",
                "Conv3x3 72->24 @128x128 output chunks",
            ],
            "rejected": ["Conv3x3 72->72 @128x128 single package"],
        },
        "end_to_end_board_status": "schedule_ready_component_board_proven_accuracy_pending_checkpoint",
    }
    plan_path = out_dir / "sdformer_aligned_hw_deployment_plan.json"
    plan_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "sdformer_aligned_hw_deployment_plan.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_shape": report["output_shape"],
                "requires_retraining": report["requires_retraining"],
                "rhb_launches": report["schedule_summary"]["rhb_launches"],
                "host_latency_ms": report["host_pytorch_latency_ms"],
                "plan": str(plan_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
