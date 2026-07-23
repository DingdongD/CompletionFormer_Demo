#!/usr/bin/env python3
from __future__ import annotations

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
    "/root/demo/artifacts/rhb_auto_config_framework/work/nlspn_strict_ref_20260723"
)
NLSPN_FEATURE_ROOT = NLSPN_WORK_ROOT / "features"
NLSPN_BOARD_ROOT = NLSPN_WORK_ROOT / "val32_board_corrected"
NLSPN_SUMMARY_JSON = NLSPN_BOARD_ROOT / "summary_aggregate.json"
DYSPN_VIS_ROOT = Path(
    "/root/demo/artifacts/visualizations/dyspn_hw128_epoch78_tailmerge7_rgbfix"
)
CSPN_LOG = Path(
    "/root/demo/artifacts/rhb_auto_config_framework/work/deployment_packages/"
    "cspn_resnettiny_hw128_w24_step8_stagewise_v3_hostsample/"
    "board_sample0_2load_revalidate_20260714.log"
)
NLSPN_LOG = Path(
    "/root/demo/artifacts/rhb_auto_config_framework/work/nlspn_strict_ref_20260723/"
    "val32_board_corrected_logs/sample00_strict_ref_board_corrected.log"
)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_NAMES = {
    "completionformer": "CompletionFormer HW128",
    "cspn": "CSPN ResNetTiny HW128",
    "nlspn": "NLSPN HW128",
    "dyspn": "DySPN HW128",
}
FALLBACK_MODEL_LATENCY_SUMMARY = {
    "completionformer": {
        "stable_latency_ms": 1492.224,
        "latency_label": "sample0 rerun with max CPU freq, mlockall, input pretouch, and GC disabled",
        "cpu_mean_ms": 35.453,
    },
    "cspn": {
        "stable_latency_ms": 2121.742,
        "latency_label": "median of accepted 2-load exact runs",
        "cpu_mean_ms": 15.102,
    },
    "nlspn": {
        "stable_latency_ms": 14202.643,
        "latency_label": "val32 mean, strict-ref 4-pack with original NLSPN propagation on RHBLite CPU",
        "cpu_mean_ms": 75.116,
    },
    "dyspn": {
        "stable_latency_ms": 4145.173,
        "latency_label": "val32 mean, tailmerge7 packer, RGB [0,1] contract",
        "cpu_mean_ms": 137.503,
    },
}


def load_model_latency_summary() -> dict:
    manifest_path = OUT_ROOT / "manifest.json"
    if not manifest_path.exists():
        return FALLBACK_MODEL_LATENCY_SUMMARY
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {}
    for model in manifest.get("models", []):
        model_id = model.get("id")
        if not model_id:
            continue
        summary[model_id] = {k: v for k, v in model.items() if k not in {"id", "name"}}
    for model_id, fallback in FALLBACK_MODEL_LATENCY_SUMMARY.items():
        summary.setdefault(model_id, fallback)
        for key, value in fallback.items():
            summary[model_id].setdefault(key, value)
    if NLSPN_SUMMARY_JSON.exists():
        nlspn_agg = json.loads(NLSPN_SUMMARY_JSON.read_text(encoding="utf-8"))
        summary.setdefault("nlspn", {})
        summary["nlspn"].update(
            {
                "stable_latency_ms": round(float(nlspn_agg["e2e_with_load_ms_mean"]), 3),
                "latency_label": "val32 mean, strict-ref 4-pack with original NLSPN propagation on RHBLite CPU",
                "rhb_total_no_load_ms": round(float(nlspn_agg["total_ms_mean"]), 3),
                "packer_load_ms": round(float(nlspn_agg["packer_load_ms_mean"]), 3),
                "wall_ms": round(float(nlspn_agg["wall_ms_mean"]), 3),
                "board_ref_l1_mean": round(float(nlspn_agg["pred_l1_mean"]), 6),
                "board_ref_rmse_mean": round(float(nlspn_agg["pred_rmse_mean"]), 6),
                "num_val_samples": int(nlspn_agg["num_samples"]),
            }
        )
    return summary


def sample_index(path: Path) -> int:
    match = re.search(r"sample(\d+)", str(path))
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def nlspn_index(path: Path) -> int:
    match = re.search(r"sample_?(\d+)", str(path))
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


def board_output_metrics(ref_pred: np.ndarray, board_pred: np.ndarray, gt: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    board_ref_err = np.abs(board_pred - ref_pred)
    metrics = {
        "output_contract": "board_pred is the final displayed/runtime output; ref_pred is comparison-only",
        "error_image_mode": "board_vs_ref",
        "abs_mean": float(np.nanmean(board_ref_err)),
        "abs_p95": float(np.nanpercentile(board_ref_err, 95)),
        "abs_max": float(np.nanmax(board_ref_err)),
        "rmse": float(np.sqrt(np.nanmean((board_pred - ref_pred) ** 2))),
        "board_ref_abs_mean": float(np.nanmean(board_ref_err)),
        "board_ref_abs_p95": float(np.nanpercentile(board_ref_err, 95)),
        "board_ref_abs_max": float(np.nanmax(board_ref_err)),
        "board_ref_rmse": float(np.sqrt(np.nanmean((board_pred - ref_pred) ** 2))),
        "ref_min": float(np.nanmin(ref_pred)),
        "ref_max": float(np.nanmax(ref_pred)),
        "board_min": float(np.nanmin(board_pred)),
        "board_max": float(np.nanmax(board_pred)),
    }
    if gt is not None:
        board_gt_err = np.abs(board_pred - gt)
        metrics.update({
            "board_gt_l1": float(np.nanmean(board_gt_err)),
            "board_gt_abs_p95": float(np.nanpercentile(board_gt_err, 95)),
            "board_gt_abs_max": float(np.nanmax(board_gt_err)),
            "board_gt_rmse": float(np.sqrt(np.nanmean((board_pred - gt) ** 2))),
        })
    return board_ref_err, metrics


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
    err, metrics = board_output_metrics(ref_pred, board_pred, gt)
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

    metrics.update({
        "board_init_min": float(np.nanmin(board_pred_init)),
        "board_init_max": float(np.nanmax(board_pred_init)),
        "runtime_mitigation": "max_cpu_freq + mlockall + input_pretouch + gc_disabled",
    })
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
    err, metrics = board_output_metrics(ref_pred, board_pred, gt)
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

    metrics.update({
        "board_init_min": float(np.nanmin(board_pred_init)),
        "board_init_max": float(np.nanmax(board_pred_init)),
        "runtime_mitigation": "2-load exact Model-Packer partition",
    })
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
    sample = f"sample{idx:02d}"
    board_npz = NLSPN_BOARD_ROOT / f"{sample}_strict_ref_board_corrected.npz"
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
    err, metrics = board_output_metrics(ref_pred, board_pred, gt)
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

    metrics.update({
        "board_init_min": float(np.nanmin(board_pred_init)),
        "board_init_max": float(np.nanmax(board_pred_init)),
        "runtime_mitigation": "4-pack strict-ref RHB Conv/head partition + board-effective output-scale correction",
    })
    metrics.update(parse_latency(NLSPN_WORK_ROOT / "val32_board_corrected_logs" / f"{sample}_strict_ref_board_corrected.log"))

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


def export_dyspn_sample(npz_path: Path) -> dict:
    idx = sample_index(npz_path.parent)
    out_dir = OUT_ROOT / f"dyspn_sample{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(npz_path, allow_pickle=True)
    rgb = denorm_rgb(z["rgb"])
    dep = squeeze_hw(z["dep"])
    gt = squeeze_hw(z["gt"])
    ref_pred = squeeze_hw(z["ref_pred"])
    board_pred = squeeze_hw(z["board_pred"])
    board_pred_init = board_pred
    err, metrics = board_output_metrics(ref_pred, board_pred, gt)
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

    metrics.update({
        "board_init_min": float(np.nanmin(board_pred_init)),
        "board_init_max": float(np.nanmax(board_pred_init)),
        "vs_gt_l1": float(np.asarray(z["vs_gt_l1"]).reshape(-1)[0]),
        "vs_gt_rmse": float(np.asarray(z["vs_gt_rmse"]).reshape(-1)[0]),
        "preprocess": "dyspn_rgb_0_1_auto_denorm",
        "runtime_mitigation": "tailmerge7 exact packer partition + clear_wr_done",
    })
    board_log = npz_path.parent / "work" / "board_runner.log"
    metrics.update(parse_latency(board_log))

    pc_path = out_dir / "point_cloud.json"
    pc_path.write_text(json.dumps(point_cloud(rgb, board_pred), separators=(",", ":")))
    meta = {
        "id": f"dyspn:{idx}",
        "index": idx,
        "model": "dyspn",
        "title": f"DySPN sample {idx}",
        "base": f"data/dyspn_sample{idx}",
        "images": images,
        "point_cloud": "point_cloud.json",
        "metrics": metrics,
        "source_npz": str(npz_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    model_latency_summary = load_model_latency_summary()
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
    nlspn_npzs = sorted(NLSPN_FEATURE_ROOT.glob("sample*_features.npz"), key=nlspn_index)
    samples.extend(export_nlspn_sample(p) for p in nlspn_npzs)
    dyspn_npzs = sorted(DYSPN_VIS_ROOT.glob("sample*/dyspn_orchestrated_board_pred_outputs.npz"), key=lambda p: sample_index(p.parent))
    samples.extend(export_dyspn_sample(p) for p in dyspn_npzs)
    manifest = {
        "name": "RHB board depth demo",
        "generated_from": "CompletionFormer, CSPN, NLSPN, and DySPN val32 board outputs",
        "point_cloud_sampling": "stride=1 full 128x128 board_pred points",
        "models": [
            {"id": model_id, "name": MODEL_NAMES[model_id], **model_latency_summary[model_id]}
            for model_id in ["completionformer", "cspn", "nlspn", "dyspn"]
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
