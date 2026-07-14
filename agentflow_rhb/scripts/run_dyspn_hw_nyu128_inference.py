#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path("/root/demo")
sys.path.insert(0, str(ROOT))

from models.dyspn_test.dyspn_hw_aligned import DySPNHWAlignedModel, load_dyspn_hw_checkpoint


DEFAULT_NYU = ROOT / "artifacts/output_completionformer_decoder_system_128x128_ckpt00059_traincalib32_20260702/nyu_val32_source_128x128.npz"
DEFAULT_OUT = ROOT / "artifacts/visualizations/dyspn_hw_aligned_nyu128"


def denorm_rgb(rgb: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    x = rgb.astype(np.float32)
    if x.min() < -0.5 or x.max() > 1.5:
        x = x * std + mean
    return np.clip(np.transpose(x, (1, 2, 0)), 0.0, 1.0)


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    mask = gt > 1e-4
    diff = pred[mask] - gt[mask]
    return {
        "l1": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def save_grid(path: Path, rgb: np.ndarray, dep: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    err = np.abs(pred - gt)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), constrained_layout=True)
    items = [
        ("rgb", denorm_rgb(rgb), None),
        ("sparse dep", dep[0], "viridis"),
        ("gt", gt[0], "turbo"),
        ("dyspn hw pred", pred[0], "turbo"),
        ("abs err", err[0], "magma"),
    ]
    for ax, (name, img, cmap) in zip(axes, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nyu-npz", default=str(DEFAULT_NYU))
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.nyu_npz)
    rgb = data["rgb"].astype(np.float32)
    dep = data["dep"].astype(np.float32)
    gt = data["gt"].astype(np.float32)
    limit = min(args.limit, rgb.shape[0])

    device = torch.device(args.device)
    torch.manual_seed(100)
    model = DySPNHWAlignedModel().to(device).eval()
    load_dyspn_hw_checkpoint(model, args.ckpt, strict=False)

    rows = ["sample,l1,rmse,max_abs,pred_min,pred_max"]
    preds = []
    with torch.no_grad():
        for idx in range(limit):
            sample_rgb = torch.from_numpy(rgb[idx : idx + 1]).to(device)
            sample_dep = torch.from_numpy(dep[idx : idx + 1]).to(device)
            output = model(sample_rgb, sample_dep)
            pred = output["pred"].detach().cpu().numpy()[0].astype(np.float32)
            preds.append(pred)
            m = depth_metrics(pred, gt[idx])
            rows.append(
                f"{idx},{m['l1']:.6f},{m['rmse']:.6f},{m['max_abs']:.6f},{float(pred.min()):.6f},{float(pred.max()):.6f}"
            )
            save_grid(
                out_dir / f"dyspn_hw_nyu128_sample{idx:02d}.png",
                rgb[idx],
                dep[idx],
                gt[idx],
                pred,
                f"sample {idx} L1={m['l1']:.4f} RMSE={m['rmse']:.4f}",
            )

    np.savez(out_dir / "dyspn_hw_nyu128_predictions.npz", pred=np.stack(preds), rgb=rgb[:limit], dep=dep[:limit], gt=gt[:limit])
    (out_dir / "dyspn_hw_nyu128_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"METRICS: {out_dir / 'dyspn_hw_nyu128_metrics.csv'}")
    print(f"PREDICTIONS: {out_dir / 'dyspn_hw_nyu128_predictions.npz'}")
    print(f"VIS_DIR: {out_dir}")


if __name__ == "__main__":
    main()

