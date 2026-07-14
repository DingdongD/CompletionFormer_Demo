#!/usr/bin/env python3
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "docs" / "data"
CSPN_VIS_ROOT = ROOT / "outputs"
NLSPN_WORK_ROOT = Path(
    "/root/demo/artifacts/rhb_auto_config_framework/work/nlspn_rebuild_current_20260713"
)
NLSPN_FEATURE_ROOT = NLSPN_WORK_ROOT / "val32_features"
NLSPN_BOARD_ROOT = NLSPN_WORK_ROOT / "val32_fix_predinit_guidance_outfit_outputs"
CSPN_LOG = ROOT / "outputs" / "cspn_unified_input" / "board_outputs" / "run_all_unified_clearwr.log"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sample_index(path: Path) -> int:
    match = re.search(r"sample(\d+)", str(path))
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def nlspn_index(path: Path) -> int:
    match = re.search(r"sample_(\d+)", str(path))
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def denorm_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if float(np.nanmin(arr)) < -0.1 or float(np.nanmax(arr)) > 1.5:
        arr = arr * STD.reshape(1, 1, 3) + MEAN.reshape(1, 1, 3)
    return np.clip(arr, 0.0, 1.0)


def squeeze_hw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(np.float32)


def save_rgb(path: Path, rgb01: np.ndarray) -> None:
    img = Image.fromarray((np.clip(rgb01, 0.0, 1.0) * 255).astype(np.uint8))
    img.save(path)


def save_map(path: Path, arr: np.ndarray, vmin=None, vmax=None, cmap_name="turbo") -> None:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if vmin is None:
        vmin = float(np.nanpercentile(arr[finite], 1)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(arr[finite], 99)) if finite.any() else 1.0
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1.0
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cm.get_cmap(cmap_name)(norm)
    img = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
    img.save(path)


def parse_latency(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    pat = re.compile(r"LATENCY\s+([^:]+):\s+([0-9.]+)\s+ms")
    total_pat = re.compile(r"LATENCY_TOTAL_MS\s+([0-9.]+)")
    summary_pat = re.compile(r"LATENCY_SUMMARY\s+(.+?)\s+([0-9.]+)$")
    wall_pat = re.compile(r"LATENCY_WALL_MS\s+([0-9.]+)")
    values = {}
    explicit_total = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            values[m.group(1).strip()] = float(m.group(2))
            continue
        m = total_pat.search(line)
        if m:
            explicit_total = float(m.group(1))
            continue
        m = summary_pat.search(line)
        if m:
            values[m.group(1).strip()] = float(m.group(2))
            continue
        m = wall_pat.search(line)
        if m:
            values["wall"] = float(m.group(1))
    if not values:
        return {}
    total = explicit_total or values.get("decoder_system_128x128_ckpt_tracked_total")
    if total is None:
        total = sum(v for k, v in values.items() if k != "decoder_system_128x128_ckpt_tracked_total")
    slowest_name, slowest_ms = max(
        ((k, v) for k, v in values.items() if k != "decoder_system_128x128_ckpt_tracked_total"),
        key=lambda kv: kv[1],
        default=("n/a", 0.0),
    )
    return {
        "latency_total_ms": total,
        "latency_slowest_op": slowest_name,
        "latency_slowest_ms": slowest_ms,
        "latencies_ms": values,
    }


def point_cloud(rgb01: np.ndarray, depth: np.ndarray) -> dict:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float32)
    mask = np.isfinite(z) & (z > 0.05)
    xx = xx[mask].astype(np.float32)
    yy = yy[mask].astype(np.float32)
    z = z[mask].astype(np.float32)
    color = rgb01[mask]
    fx = fy = float(max(h, w) * 0.9)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    x = (xx - cx) * z / fx
    y = -(yy - cy) * z / fy
    colors = ["#%02x%02x%02x" % tuple((np.clip(c, 0.0, 1.0) * 255).astype(np.uint8)) for c in color]
    return {
        "x": np.round(x, 5).tolist(),
        "y": np.round(y, 5).tolist(),
        "z": np.round(z, 5).tolist(),
        "color": colors,
        "count": int(z.size),
    }


def export_sample(npz_path: Path) -> dict:
    idx = sample_index(npz_path)
    out_dir = OUT_ROOT / f"sample{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(npz_path)
    rgb = denorm_rgb(z["rgb"])
    dep = squeeze_hw(z["dep"])
    gt = squeeze_hw(z["gt"])
    ref_pred = squeeze_hw(z["ref_pred"])
    board_pred = squeeze_hw(z["board_pred"])
    board_pred_init = squeeze_hw(z["board_pred_init"])
    err = np.abs(board_pred - ref_pred)
    depth_vmin = float(min(np.nanpercentile(ref_pred, 1), np.nanpercentile(board_pred, 1)))
    depth_vmax = float(max(np.nanpercentile(ref_pred, 99), np.nanpercentile(board_pred, 99)))
    sparse_vis = np.where(dep > 0, dep, np.nan)

    images = {
        "rgb": "rgb.png",
        "sparse_depth": "sparse_depth.png",
        "gt": "gt.png",
        "ref_pred": "ref_pred.png",
        "board_pred": "board_pred.png",
        "abs_error": "abs_error.png",
    }
    save_rgb(out_dir / images["rgb"], rgb)
    save_map(out_dir / images["sparse_depth"], sparse_vis, depth_vmin, depth_vmax)
    save_map(out_dir / images["gt"], gt, depth_vmin, depth_vmax)
    save_map(out_dir / images["ref_pred"], ref_pred, depth_vmin, depth_vmax)
    save_map(out_dir / images["board_pred"], board_pred, depth_vmin, depth_vmax)
    save_map(out_dir / images["abs_error"], err, 0.0, max(0.1, float(np.nanpercentile(err, 99))), "magma")

    metrics = {
        "abs_mean": float(np.nanmean(err)),
        "abs_p95": float(np.nanpercentile(err, 95)),
        "abs_max": float(np.nanmax(err)),
        "rmse": float(np.sqrt(np.nanmean((board_pred - ref_pred) ** 2))),
        "ref_min": float(np.nanmin(ref_pred)),
        "ref_max": float(np.nanmax(ref_pred)),
        "board_min": float(np.nanmin(board_pred)),
        "board_max": float(np.nanmax(board_pred)),
        "board_init_min": float(np.nanmin(board_pred_init)),
        "board_init_max": float(np.nanmax(board_pred_init)),
    }
    metrics.update(parse_latency(npz_path.parent / f"board_val{idx}_convonlycf_hostsigmoid.log"))

    pc_path = out_dir / "point_cloud.json"
    pc_path.write_text(json.dumps(point_cloud(rgb, board_pred), separators=(",", ":")))
    meta = {
        "index": idx,
        "title": f"sample {idx}",
        "base": f"data/sample{idx}",
        "images": images,
        "point_cloud": "point_cloud.json",
        "metrics": metrics,
        "source_npz": str(npz_path.relative_to(ROOT)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def export_cspn_sample(npz_path: Path) -> dict:
    idx = sample_index(npz_path)
    out_dir = OUT_ROOT / f"cspn_sample{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(npz_path)
    rgb = np.asarray(z["rgb"], dtype=np.float32)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    dep = squeeze_hw(z["sparse"])
    gt = squeeze_hw(z["gt"])
    ref_pred = squeeze_hw(z["ref_pred"])
    board_pred = squeeze_hw(z["board_pred"])
    board_pred_init = squeeze_hw(z["board_raw"]) if "board_raw" in z.files else board_pred
    err = np.abs(board_pred - ref_pred)
    depth_vmin = float(min(np.nanpercentile(ref_pred, 1), np.nanpercentile(board_pred, 1)))
    depth_vmax = float(max(np.nanpercentile(ref_pred, 99), np.nanpercentile(board_pred, 99)))
    sparse_vis = np.where(dep > 0, dep, np.nan)

    images = {
        "rgb": "rgb.png",
        "sparse_depth": "sparse_depth.png",
        "gt": "gt.png",
        "ref_pred": "ref_pred.png",
        "board_pred": "board_pred.png",
        "abs_error": "abs_error.png",
    }
    save_rgb(out_dir / images["rgb"], rgb)
    save_map(out_dir / images["sparse_depth"], sparse_vis, depth_vmin, depth_vmax)
    save_map(out_dir / images["gt"], gt, depth_vmin, depth_vmax)
    save_map(out_dir / images["ref_pred"], ref_pred, depth_vmin, depth_vmax)
    save_map(out_dir / images["board_pred"], board_pred, depth_vmin, depth_vmax)
    save_map(out_dir / images["abs_error"], err, 0.0, max(0.1, float(np.nanpercentile(err, 99))), "magma")

    metrics = {
        "abs_mean": float(np.nanmean(err)),
        "abs_p95": float(np.nanpercentile(err, 95)),
        "abs_max": float(np.nanmax(err)),
        "rmse": float(np.sqrt(np.nanmean((board_pred - ref_pred) ** 2))),
        "ref_min": float(np.nanmin(ref_pred)),
        "ref_max": float(np.nanmax(ref_pred)),
        "board_min": float(np.nanmin(board_pred)),
        "board_max": float(np.nanmax(board_pred)),
        "board_init_min": float(np.nanmin(board_pred_init)),
        "board_init_max": float(np.nanmax(board_pred_init)),
    }
    metrics.update(parse_latency(CSPN_LOG))

    pc_path = out_dir / "point_cloud.json"
    pc_path.write_text(json.dumps(point_cloud(rgb, board_pred), separators=(",", ":")))
    meta = {
        "id": f"cspn:{idx}",
        "index": idx,
        "model": "cspn",
        "title": f"CSPN sample {idx}",
        "base": f"data/cspn_sample{idx}",
        "images": images,
        "point_cloud": "point_cloud.json",
        "metrics": metrics,
        "source_npz": str(npz_path.relative_to(ROOT)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def export_nlspn_sample(feature_npz: Path) -> dict:
    idx = nlspn_index(feature_npz)
    board_npz = NLSPN_BOARD_ROOT / f"sample_{idx}_fix_predinit_guidance_outfit.npz"
    if not board_npz.exists():
        raise FileNotFoundError(board_npz)
    out_dir = OUT_ROOT / f"nlspn_sample{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = np.load(feature_npz, allow_pickle=True)
    board = np.load(board_npz, allow_pickle=True)
    rgb = denorm_rgb(feat["rgb"])
    dep = squeeze_hw(feat["sparse"])
    gt = squeeze_hw(feat["gt"])
    ref_pred = squeeze_hw(feat["ref_pred"])
    board_pred = squeeze_hw(board["pred"])
    board_pred_init = squeeze_hw(board["pred_init"])
    err = np.abs(board_pred - ref_pred)
    depth_vmin = float(min(np.nanpercentile(ref_pred, 1), np.nanpercentile(board_pred, 1)))
    depth_vmax = float(max(np.nanpercentile(ref_pred, 99), np.nanpercentile(board_pred, 99)))
    sparse_vis = np.where(dep > 0, dep, np.nan)

    images = {
        "rgb": "rgb.png",
        "sparse_depth": "sparse_depth.png",
        "gt": "gt.png",
        "ref_pred": "ref_pred.png",
        "board_pred": "board_pred.png",
        "abs_error": "abs_error.png",
    }
    save_rgb(out_dir / images["rgb"], rgb)
    save_map(out_dir / images["sparse_depth"], sparse_vis, depth_vmin, depth_vmax)
    save_map(out_dir / images["gt"], gt, depth_vmin, depth_vmax)
    save_map(out_dir / images["ref_pred"], ref_pred, depth_vmin, depth_vmax)
    save_map(out_dir / images["board_pred"], board_pred, depth_vmin, depth_vmax)
    save_map(out_dir / images["abs_error"], err, 0.0, max(0.1, float(np.nanpercentile(err, 99))), "magma")

    metrics = {
        "abs_mean": float(np.nanmean(err)),
        "abs_p95": float(np.nanpercentile(err, 95)),
        "abs_max": float(np.nanmax(err)),
        "rmse": float(np.sqrt(np.nanmean((board_pred - ref_pred) ** 2))),
        "ref_min": float(np.nanmin(ref_pred)),
        "ref_max": float(np.nanmax(ref_pred)),
        "board_min": float(np.nanmin(board_pred)),
        "board_max": float(np.nanmax(board_pred)),
        "board_init_min": float(np.nanmin(board_pred_init)),
        "board_init_max": float(np.nanmax(board_pred_init)),
    }

    pc_path = out_dir / "point_cloud.json"
    pc_path.write_text(json.dumps(point_cloud(rgb, board_pred), separators=(",", ":")))
    meta = {
        "id": f"nlspn:{idx}",
        "index": idx,
        "model": "nlspn",
        "title": f"NLSPN sample {idx}",
        "base": f"data/nlspn_sample{idx}",
        "images": images,
        "point_cloud": "point_cloud.json",
        "metrics": metrics,
        "source_npz": str(feature_npz),
        "board_npz": str(board_npz),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    npzs = sorted(ROOT.glob("outputs/sample*/nyu_val*_ref_vs_board_convonlycf_hostsigmoid.npz"), key=sample_index)
    if not npzs:
        raise SystemExit("No board visualization NPZ files found under outputs/sample*/")
    samples = [export_sample(p) for p in npzs]
    for sample in samples:
        sample["id"] = f"completionformer:{sample['index']}"
        sample["model"] = "completionformer"
        sample["title"] = f"CompletionFormer sample {sample['index']}"
        (OUT_ROOT / f"sample{sample['index']}" / "meta.json").write_text(json.dumps(sample, indent=2))
    cspn_npzs = sorted(CSPN_VIS_ROOT.glob("cspn_sample*/cspn_val*_padded16_board_pred_outputs.npz"), key=sample_index)
    samples.extend(export_cspn_sample(p) for p in cspn_npzs)
    nlspn_npzs = sorted(NLSPN_FEATURE_ROOT.glob("sample_*.npz"), key=nlspn_index)
    samples.extend(export_nlspn_sample(p) for p in nlspn_npzs)
    manifest = {
        "name": "RHB board depth demo",
        "generated_from": "CompletionFormer outputs/sample*/ plus CSPN unified and NLSPN split-dec5 val32 board outputs",
        "point_cloud_sampling": "stride=1 full 128x128 board_pred points",
        "models": [
            {"id": "completionformer", "name": "CompletionFormer HW128"},
            {"id": "cspn", "name": "CSPN ResNetTiny HW128"},
            {"id": "nlspn", "name": "NLSPN HW128"},
        ],
        "samples": [
            {
                "id": s["id"],
                "index": s["index"],
                "model": s["model"],
                "title": s["title"],
                "base": s["base"],
                "metrics": s["metrics"],
            }
            for s in samples
        ],
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {len(samples)} samples to {OUT_ROOT}")


if __name__ == "__main__":
    main()
