#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/root/demo")
DEFAULT_SUBSET = ROOT / "artifacts/nyu_val_representative_32_128x128"


def center_crop_tensor(x: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return x[..., top : top + crop_h, left : left + crop_w]


def load_sample(path: Path, height: int, width: int, crop_h: int, crop_w: int, normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        rgb = f["rgb"][:].astype(np.float32) / 255.0
        dep = f["depth"][:].astype(np.float32)[None]

    rgb_t = torch.from_numpy(rgb)[None]
    dep_t = torch.from_numpy(dep)[None]
    rgb_t = F.interpolate(rgb_t, size=(height, width), mode="bilinear", align_corners=False)
    dep_t = F.interpolate(dep_t, size=(height, width), mode="nearest")
    rgb_t = center_crop_tensor(rgb_t, crop_h, crop_w)
    dep_t = center_crop_tensor(dep_t, crop_h, crop_w)
    rgb_arr = rgb_t[0].numpy().astype(np.float32)
    dep_arr = dep_t[0].numpy().astype(np.float32)
    if normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        rgb_arr = (rgb_arr - mean) / std
    return rgb_arr, dep_arr


def sparse_depth(dep: np.ndarray, num_sample: int, seed: int) -> np.ndarray:
    flat = dep.reshape(-1)
    valid = np.flatnonzero(flat > 1e-4)
    rng = np.random.default_rng(seed)
    if num_sample > 0 and len(valid) > num_sample:
        valid = rng.choice(valid, size=num_sample, replace=False)
    mask = np.zeros_like(flat, dtype=np.float32)
    mask[valid] = 1.0
    return (flat * mask).reshape(dep.shape).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", default=str(DEFAULT_SUBSET))
    parser.add_argument("--split-json", default="")
    parser.add_argument("--out", default=str(ROOT / "artifacts/rhb_auto_config_framework/work/dyspn_nyu_val32_128x128.npz"))
    parser.add_argument("--mode", default="val", choices=["train", "val", "test"])
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--crop-height", type=int, default=128)
    parser.add_argument("--crop-width", type=int, default=128)
    parser.add_argument("--num-sample", type=int, default=500)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7240)
    parser.add_argument("--imagenet-normalize", action="store_true")
    args = parser.parse_args()

    subset = Path(args.subset_root)
    split_path = Path(args.split_json) if args.split_json else next(subset.glob("*_split.json"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    entries = split[args.mode] or split.get("val", []) or split.get("test", []) or split.get("train", [])
    entries = entries[: args.limit]

    rgbs, deps, gts = [], [], []
    for idx, item in enumerate(entries):
        rel = item["filename"]
        path = subset / "nyudepth_hdf5" / rel
        rgb, gt = load_sample(path, args.height, args.width, args.crop_height, args.crop_width, args.imagenet_normalize)
        dep = sparse_depth(gt, args.num_sample, args.seed + idx)
        rgbs.append(rgb)
        deps.append(dep)
        gts.append(gt)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rgb=np.stack(rgbs), dep=np.stack(deps), gt=np.stack(gts))
    print(f"OUT: {out}")
    print(f"rgb: {np.stack(rgbs).shape} dep: {np.stack(deps).shape} gt: {np.stack(gts).shape}")
    print(f"normalize: {'imagenet' if args.imagenet_normalize else 'raw_0_1'}")


if __name__ == "__main__":
    main()

