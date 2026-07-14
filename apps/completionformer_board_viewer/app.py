#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DEFAULT_PORTABLE_DIR = Path(
    "/root/demo/artifacts/successful_pipelines/"
    "completionformer_hw128_ckpt00059_rhb_convonlycf_20260702/portable_runtime"
)
PORTABLE_DIR = Path(os.environ.get("CF_PORTABLE_DIR", DEFAULT_PORTABLE_DIR)).resolve()
NLSPN_WORK_DIR = Path(
    "/root/demo/artifacts/rhb_auto_config_framework/work/nlspn_rebuild_current_20260713"
)
CSPN_PACKAGE_DIR = Path(
    os.environ.get(
        "CSPN_PACKAGE_DIR",
        "/root/demo/artifacts/rhb_auto_config_framework/work/deployment_packages/"
        "cspn_resnettiny_hw128_w24_step8_stagewise_v3_hostsample",
    )
).resolve()
CSPN_VIS_ROOT = Path(os.environ.get("CSPN_VIS_ROOT", PORTABLE_DIR / "outputs")).resolve()
CSPN_VAL32_BOARD_DIR = Path(
    os.environ.get(
        "CSPN_VAL32_BOARD_DIR",
        str(PORTABLE_DIR / "outputs" / "cspn_unified_input" / "board_outputs"),
    )
).resolve()
MODELS = {
    "completionformer": {
        "key": "completionformer",
        "label": "CompletionFormer HW128",
        "description": "ckpt00059, RHB conv-only confidence head, host sigmoid",
        "portable_dir": PORTABLE_DIR,
        "dataset_npz": PORTABLE_DIR / "data" / "nyu_val32_source_128x128.npz",
        "output_dir": PORTABLE_DIR / "outputs",
        "run_script": PORTABLE_DIR / "run_board_single_sample.sh",
        "kind": "completionformer",
    },
    "nlspn": {
        "key": "nlspn",
        "label": "NLSPN HW128",
        "description": "split-dec5 scale-contract board outputs, val32",
        "portable_dir": NLSPN_WORK_DIR,
        "feature_dir": NLSPN_WORK_DIR / "val32_features",
        "output_dir": NLSPN_WORK_DIR / "val32_fix_predinit_guidance_outfit_outputs",
        "metrics_csv": NLSPN_WORK_DIR / "val32_fix_predinit_guidance_outfit_outputs" / "metrics.csv",
        "run_script": None,
        "kind": "nlspn",
    },
    "cspn": {
        "key": "cspn",
        "label": "CSPN ResNetTiny HW128",
        "description": "stagewise scale-aware RHB deployment with padded16 final depth head",
        "portable_dir": CSPN_PACKAGE_DIR,
        "vis_root": CSPN_VIS_ROOT,
        "board_dir": CSPN_VAL32_BOARD_DIR,
        "run_script": None,
        "kind": "cspn",
    },
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def model_config(model_key) -> dict:
    key = model_key or "completionformer"
    if key not in MODELS:
        raise KeyError(f"Unknown model: {key}")
    return MODELS[key]


def completionformer_sample_paths(cfg: dict, index: int) -> dict[str, Path]:
    sample_dir = cfg["output_dir"] / f"sample{index}"
    tag = f"nyu_val{index}_ref_vs_board_convonlycf_hostsigmoid"
    return {
        "sample_dir": sample_dir,
        "feature_npz": sample_dir / f"nyu_val{index}_features_ckpt00059.npz",
        "board_npz": sample_dir / f"board_val{index}_convonlycf_hostsigmoid_outputs.npz",
        "vis_npz": sample_dir / f"{tag}.npz",
        "vis_png": sample_dir / f"{tag}.png",
        "log": sample_dir / f"board_val{index}_convonlycf_hostsigmoid.log",
    }


def nlspn_sample_paths(cfg: dict, index: int) -> dict[str, Path]:
    return {
        "feature_npz": cfg["feature_dir"] / f"sample_{index}.npz",
        "board_npz": cfg["output_dir"] / f"sample_{index}_fix_predinit_guidance_outfit.npz",
        "vis_npz": cfg["output_dir"] / f"sample_{index}_fix_predinit_guidance_outfit.npz",
        "vis_png": NLSPN_WORK_DIR / "val32_fix_predinit_guidance_outfit_viz" / "nlspn_val32_ref_board_error_contact_sheet.png",
        "log": cfg["output_dir"] / f"sample_{index}.log",
    }


def cspn_sample_paths(cfg: dict, index: int) -> dict[str, Path]:
    return {
        "vis_npz": cfg["vis_root"] / f"cspn_sample{index}" / f"cspn_val{index}_padded16_board_pred_outputs.npz",
        "board_npz": cfg["board_dir"] / f"cspn_val{index:02d}_board_padded16_clearwr_run1.npz",
        "vis_png": cfg["vis_root"] / "docs" / "data" / f"cspn_sample{index}" / "board_pred.png",
        "log": cfg["board_dir"] / "run_all_unified_clearwr.log",
    }


def sample_paths(model_key, index: int) -> dict:
    cfg = model_config(model_key)
    if cfg["kind"] == "completionformer":
        return completionformer_sample_paths(cfg, index)
    if cfg["kind"] == "nlspn":
        return nlspn_sample_paths(cfg, index)
    if cfg["kind"] == "cspn":
        return cspn_sample_paths(cfg, index)
    raise ValueError(f"Unsupported model kind: {cfg['kind']}")


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


def png_data_uri_rgb(rgb01: np.ndarray) -> str:
    img = Image.fromarray((np.clip(rgb01, 0.0, 1.0) * 255).astype(np.uint8))
    return encode_png(img)


def png_data_uri_map(arr: np.ndarray, vmin=None, vmax=None, cmap_name="turbo") -> str:
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
    return encode_png(img)


def encode_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def point_cloud_payload(rgb01: np.ndarray, depth: np.ndarray, stride: int = 1, max_points: int = 20000) -> dict:
    depth = np.asarray(depth, dtype=np.float32)
    rgb = np.asarray(rgb01)
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    color = rgb[::stride, ::stride]
    mask = np.isfinite(z) & (z > 0.05)
    yy = yy[mask].astype(np.float32)
    xx = xx[mask].astype(np.float32)
    z = z[mask].astype(np.float32)
    color = color[mask]
    if z.size > max_points:
        keep = np.linspace(0, z.size - 1, max_points).astype(np.int64)
        xx, yy, z, color = xx[keep], yy[keep], z[keep], color[keep]
    fx = fy = float(max(h, w) * 0.9)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    x = (xx - cx) * z / fx
    y = -(yy - cy) * z / fy
    colors = [
        f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
        for r, g, b in np.clip(color, 0.0, 1.0)
    ]
    return {
        "x": x.round(5).tolist(),
        "y": y.round(5).tolist(),
        "z": z.round(5).tolist(),
        "color": colors,
        "count": int(z.size),
    }



def parse_latency_metrics(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    latencies = {}
    pattern = re.compile(r"LATENCY\s+([^:]+):\s+([0-9.]+)\s+ms")
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            latencies[match.group(1).strip()] = float(match.group(2))
    if not latencies:
        return {}
    total = latencies.get("decoder_system_128x128_ckpt_tracked_total")
    if total is None:
        total = sum(v for k, v in latencies.items() if k != "decoder_system_128x128_ckpt_tracked_total")
    slowest_name = None
    slowest_ms = None
    for name, value in latencies.items():
        if name == "decoder_system_128x128_ckpt_tracked_total":
            continue
        if slowest_ms is None or value > slowest_ms:
            slowest_name = name
            slowest_ms = value
    return {
        "latency_total_ms": float(total),
        "latency_slowest_ms": float(slowest_ms) if slowest_ms is not None else None,
        "latency_slowest_op": slowest_name or "n/a",
        "latencies_ms": latencies,
    }

def load_completionformer_arrays(paths):
    if not paths["vis_npz"].exists():
        raise FileNotFoundError(f"Board visualization NPZ does not exist: {paths['vis_npz']}")
    z = np.load(paths["vis_npz"])
    return (
        denorm_rgb(z["rgb"]),
        squeeze_hw(z["dep"]),
        squeeze_hw(z["gt"]),
        squeeze_hw(z["ref_pred"]),
        squeeze_hw(z["board_pred"]),
        squeeze_hw(z["board_pred_init"]),
    )


def load_nlspn_arrays(paths):
    if not paths["feature_npz"].exists():
        raise FileNotFoundError(f"NLSPN feature NPZ does not exist: {paths['feature_npz']}")
    if not paths["board_npz"].exists():
        raise FileNotFoundError(f"NLSPN board NPZ does not exist: {paths['board_npz']}")
    feat = np.load(paths["feature_npz"], allow_pickle=True)
    board = np.load(paths["board_npz"], allow_pickle=True)
    return (
        denorm_rgb(feat["rgb"]),
        squeeze_hw(feat["sparse"]),
        squeeze_hw(feat["gt"]),
        squeeze_hw(feat["ref_pred"]),
        squeeze_hw(board["pred"]),
        squeeze_hw(board["pred_init"]),
    )


def load_cspn_arrays(paths):
    if not paths["vis_npz"].exists():
        raise FileNotFoundError(f"CSPN visualization NPZ does not exist: {paths['vis_npz']}")
    z = np.load(paths["vis_npz"], allow_pickle=True)
    return (
        denorm_rgb(z["rgb"]),
        squeeze_hw(z["sparse"]),
        squeeze_hw(z["gt"]),
        squeeze_hw(z["ref_pred"]),
        squeeze_hw(z["board_pred"]),
        squeeze_hw(z["board_raw"]) if "board_raw" in z.files else squeeze_hw(z["board_pred"]),
    )


def load_csv_metrics(path: Path, index: int) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            sample_value = row.get("sample") or row.get("index") or row.get("sample_idx")
            if sample_value is None:
                continue
            try:
                if int(float(sample_value)) != index:
                    continue
            except ValueError:
                continue
            metrics = {}
            for key, value in row.items():
                if key in {"sample", "index", "sample_idx"} or value in {None, ""}:
                    continue
                try:
                    metrics[f"csv_{key}"] = float(value)
                except ValueError:
                    metrics[f"csv_{key}"] = value
            return metrics
    return {}


def load_sample_payload(index: int, model_key=None) -> dict:
    cfg = model_config(model_key)
    paths = sample_paths(cfg["key"], index)
    if cfg["kind"] == "completionformer":
        rgb01, dep, gt, ref_pred, board_pred, board_pred_init = load_completionformer_arrays(paths)
    elif cfg["kind"] == "nlspn":
        rgb01, dep, gt, ref_pred, board_pred, board_pred_init = load_nlspn_arrays(paths)
    elif cfg["kind"] == "cspn":
        rgb01, dep, gt, ref_pred, board_pred, board_pred_init = load_cspn_arrays(paths)
    else:
        raise ValueError(f"Unsupported model kind: {cfg['kind']}")
    err = np.abs(board_pred - ref_pred)
    depth_vmin = float(min(np.nanpercentile(ref_pred, 1), np.nanpercentile(board_pred, 1)))
    depth_vmax = float(max(np.nanpercentile(ref_pred, 99), np.nanpercentile(board_pred, 99)))
    sparse_vis = np.where(dep > 0, dep, np.nan)
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
    metrics.update(parse_latency_metrics(paths["log"]))
    if cfg.get("metrics_csv"):
        metrics.update(load_csv_metrics(cfg["metrics_csv"], index))
    return {
        "model": {
            "key": cfg["key"],
            "label": cfg["label"],
            "description": cfg["description"],
        },
        "index": index,
        "portable_dir": str(cfg["portable_dir"]),
        "paths": {k: str(v) for k, v in paths.items()},
        "metrics": metrics,
        "images": {
            "rgb": png_data_uri_rgb(rgb01),
            "sparse_depth": png_data_uri_map(sparse_vis, depth_vmin, depth_vmax),
            "gt": png_data_uri_map(gt, depth_vmin, depth_vmax),
            "ref_pred": png_data_uri_map(ref_pred, depth_vmin, depth_vmax),
            "board_pred": png_data_uri_map(board_pred, depth_vmin, depth_vmax),
            "abs_error": png_data_uri_map(err, 0.0, max(0.1, float(np.nanpercentile(err, 99))), "magma"),
        },
        "point_cloud": point_cloud_payload(rgb01, board_pred),
    }


def run_board_sample(index: int, model_key=None) -> dict:
    cfg = model_config(model_key)
    script = cfg.get("run_script")
    if script is None:
        raise RuntimeError(f"{cfg['label']} currently exposes precomputed board outputs only; no single-sample board runner is packaged.")
    if not script.exists():
        raise FileNotFoundError(f"Missing board runner: {script}")
    env = os.environ.copy()
    env.setdefault("BOARD", "root@192.168.115.122")
    env.setdefault("BOARD_PASS", "root")
    started = time.time()
    proc = subprocess.run(
        [str(script), str(index)],
        cwd=str(cfg["portable_dir"]),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(os.environ.get("CF_BOARD_TIMEOUT", "900")),
    )
    elapsed = time.time() - started
    return {
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "output_tail": "\n".join(proc.stdout.splitlines()[-80:]),
    }


def model_public_info(cfg: dict) -> dict:
    return {
        "key": cfg["key"],
        "label": cfg["label"],
        "description": cfg["description"],
        "portable_dir": str(cfg["portable_dir"]),
        "can_run_board": cfg.get("run_script") is not None and cfg["run_script"].exists(),
    }


def list_models() -> dict:
    return {"models": [model_public_info(cfg) for cfg in MODELS.values()]}


def list_samples(model_key=None) -> dict:
    cfg = model_config(model_key)
    count = 32
    if cfg["kind"] == "completionformer" and cfg["dataset_npz"].exists():
        with np.load(cfg["dataset_npz"]) as z:
            first = z[z.files[0]]
            count = int(first.shape[0]) if first.ndim > 0 else 32
    elif cfg["kind"] == "nlspn":
        count = len(list(cfg["feature_dir"].glob("sample_*.npz"))) if cfg["feature_dir"].exists() else 0
    elif cfg["kind"] == "cspn":
        count = len(list(cfg["vis_root"].glob("cspn_sample*/cspn_val*_padded16_board_pred_outputs.npz"))) if cfg["vis_root"].exists() else 0
    samples = []
    for i in range(count):
        paths = sample_paths(cfg["key"], i)
        samples.append({
            "index": i,
            "has_board_output": paths["vis_npz"].exists(),
            "vis_npz": str(paths["vis_npz"]),
            "vis_png": str(paths["vis_png"]),
        })
    return {"model": model_public_info(cfg), "portable_dir": str(cfg["portable_dir"]), "samples": samples}


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        if self.path == "/" or self.path.startswith("/static/"):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        if path == "/" or path.startswith("/static/"):
            if path == "/":
                return str(STATIC_DIR / "index.html")
            return str(APP_DIR / path.lstrip("/"))
        return str(STATIC_DIR / "index.html")

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        model_key = (query.get("model") or [None])[0]
        if parsed.path == "/api/models":
            return self.write_json(list_models())
        if parsed.path == "/api/samples":
            try:
                return self.write_json(list_samples(model_key))
            except Exception as exc:
                return self.write_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        if parsed.path.startswith("/api/sample/"):
            try:
                index = int(parsed.path.rsplit("/", 1)[-1])
                return self.write_json(load_sample_payload(index, model_key))
            except Exception as exc:
                return self.write_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        if parsed.path == "/api/tof/status":
            return self.write_json({
                "status": "reserved",
                "message": "ToF adapter hook is reserved; no live device backend is attached in this app.",
            })
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        model_key = (query.get("model") or [None])[0]
        if parsed.path.startswith("/api/run/"):
            try:
                index = int(parsed.path.rsplit("/", 1)[-1])
                run = run_board_sample(index, model_key)
                payload = {"run": run}
                if run["returncode"] == 0:
                    payload["sample"] = load_sample_payload(index, model_key)
                status = HTTPStatus.OK if run["returncode"] == 0 else HTTPStatus.INTERNAL_SERVER_ERROR
                return self.write_json(payload, status)
            except Exception as exc:
                return self.write_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        if parsed.path == "/api/upload-rgbd":
            return self.write_json({
                "status": "staged",
                "message": "RGBD upload endpoint is reserved. Current board runner consumes the packaged NYU 128x128 NPZ samples.",
            })
        return self.write_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)

    def write_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CompletionFormer board viewer: http://{args.host}:{args.port}")
    print(f"Portable runtime: {PORTABLE_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
