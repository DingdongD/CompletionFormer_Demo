#!/usr/bin/env python3
import argparse
import base64
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
DATASET_NPZ = PORTABLE_DIR / "data" / "nyu_val32_source_128x128.npz"
OUTPUT_DIR = PORTABLE_DIR / "outputs"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sample_paths(index: int) -> dict[str, Path]:
    sample_dir = OUTPUT_DIR / f"sample{index}"
    tag = f"nyu_val{index}_ref_vs_board_convonlycf_hostsigmoid"
    return {
        "sample_dir": sample_dir,
        "feature_npz": sample_dir / f"nyu_val{index}_features_ckpt00059.npz",
        "board_npz": sample_dir / f"board_val{index}_convonlycf_hostsigmoid_outputs.npz",
        "vis_npz": sample_dir / f"{tag}.npz",
        "vis_png": sample_dir / f"{tag}.png",
        "log": sample_dir / f"board_val{index}_convonlycf_hostsigmoid.log",
    }


def denorm_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
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

def load_sample_payload(index: int) -> dict:
    paths = sample_paths(index)
    if not paths["vis_npz"].exists():
        raise FileNotFoundError(f"Board visualization NPZ does not exist: {paths['vis_npz']}")
    z = np.load(paths["vis_npz"])
    rgb01 = denorm_rgb(z["rgb"])
    dep = squeeze_hw(z["dep"])
    gt = squeeze_hw(z["gt"])
    ref_pred = squeeze_hw(z["ref_pred"])
    board_pred = squeeze_hw(z["board_pred"])
    board_pred_init = squeeze_hw(z["board_pred_init"])
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
    return {
        "index": index,
        "portable_dir": str(PORTABLE_DIR),
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


def run_board_sample(index: int) -> dict:
    script = PORTABLE_DIR / "run_board_single_sample.sh"
    if not script.exists():
        raise FileNotFoundError(f"Missing board runner: {script}")
    env = os.environ.copy()
    env.setdefault("BOARD", "root@192.168.115.122")
    env.setdefault("BOARD_PASS", "root")
    started = time.time()
    proc = subprocess.run(
        [str(script), str(index)],
        cwd=str(PORTABLE_DIR),
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


def list_samples() -> dict:
    count = 32
    if DATASET_NPZ.exists():
        with np.load(DATASET_NPZ) as z:
            first = z[z.files[0]]
            count = int(first.shape[0]) if first.ndim > 0 else 32
    samples = []
    for i in range(count):
        paths = sample_paths(i)
        samples.append({
            "index": i,
            "has_board_output": paths["vis_npz"].exists(),
            "vis_npz": str(paths["vis_npz"]),
            "vis_png": str(paths["vis_png"]),
        })
    return {"portable_dir": str(PORTABLE_DIR), "samples": samples}


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        if path == "/" or path.startswith("/static/"):
            if path == "/":
                return str(STATIC_DIR / "index.html")
            return str(APP_DIR / path.lstrip("/"))
        return str(STATIC_DIR / "index.html")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/samples":
            return self.write_json(list_samples())
        if parsed.path.startswith("/api/sample/"):
            try:
                index = int(parsed.path.rsplit("/", 1)[-1])
                return self.write_json(load_sample_payload(index))
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
        if parsed.path.startswith("/api/run/"):
            try:
                index = int(parsed.path.rsplit("/", 1)[-1])
                run = run_board_sample(index)
                payload = {"run": run}
                if run["returncode"] == 0:
                    payload["sample"] = load_sample_payload(index)
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
