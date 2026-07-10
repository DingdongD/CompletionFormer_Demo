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
DEFAULT_CSPN_PACKAGE_DIR = Path(
    "/root/demo/artifacts/rhb_auto_config_framework/work/deployment_packages/"
    "cspn_resnettiny_hw128_w24_step8_stagewise_v3_hostsample"
)
CSPN_PACKAGE_DIR = Path(os.environ.get("CSPN_PACKAGE_DIR", DEFAULT_CSPN_PACKAGE_DIR)).resolve()
CSPN_VIS_ROOT = Path(os.environ.get("CSPN_VIS_ROOT", PORTABLE_DIR / "outputs")).resolve()
CSPN_VAL32_INPUT_DIR = Path(
    os.environ.get(
        "CSPN_VAL32_INPUT_DIR",
        str(PORTABLE_DIR / "outputs" / "cspn_unified_input" / "inputs"),
    )
).resolve()
CSPN_VAL32_BOARD_DIR = Path(
    os.environ.get(
        "CSPN_VAL32_BOARD_DIR",
        str(PORTABLE_DIR / "outputs" / "cspn_unified_input" / "board_outputs"),
    )
).resolve()
CSPN_EXPORT_SCRIPT = PORTABLE_DIR / "scripts" / "export_cspn_app_val_outputs.py"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_model_id(model_id=None) -> str:
    model_id = (model_id or "completionformer").strip().lower()
    if model_id in {"cf", "completionformer"}:
        return "completionformer"
    if model_id in {"cspn", "cspn-tiny", "cspn_resnettiny"}:
        return "cspn"
    raise ValueError(f"Unknown model: {model_id}")


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
    summary_pattern = re.compile(r"LATENCY_SUMMARY\s+(.+?)\s+([0-9.]+)$")
    wall_pattern = re.compile(r"LATENCY_WALL_MS\s+([0-9.]+)")
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            latencies[match.group(1).strip()] = float(match.group(2))
            continue
        match = summary_pattern.search(line)
        if match:
            latencies[match.group(1).strip()] = float(match.group(2))
            continue
        match = wall_pattern.search(line)
        if match:
            latencies["wall"] = float(match.group(1))
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


def load_completionformer_payload(index: int) -> dict:
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


def load_cspn_payload(index: int) -> dict:
    vis_npz = CSPN_VIS_ROOT / f"cspn_sample{index}" / f"cspn_val{index}_padded16_board_pred_outputs.npz"
    board_npz = CSPN_VAL32_BOARD_DIR / f"cspn_val{index:02d}_board_padded16_clearwr_run1.npz"
    log_path = CSPN_VAL32_BOARD_DIR / "run_all_unified_clearwr.log"
    if not vis_npz.exists():
        raise FileNotFoundError(f"CSPN visualization NPZ does not exist: {vis_npz}")
    z = np.load(vis_npz)
    rgb01 = np.asarray(z["rgb"], dtype=np.float32)
    if rgb01.ndim == 4:
        rgb01 = rgb01[0]
    if rgb01.shape[0] == 3:
        rgb01 = np.transpose(rgb01, (1, 2, 0))
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    dep = squeeze_hw(z["sparse"])
    gt = squeeze_hw(z["gt"])
    ref_pred = squeeze_hw(z["ref_pred"])
    board_pred = squeeze_hw(z["board_pred"])
    board_pred_init = squeeze_hw(z["board_raw"]) if "board_raw" in z.files else board_pred
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
    metrics.update(parse_latency_metrics(log_path))
    return {
        "index": index,
        "model": "cspn",
        "portable_dir": str(CSPN_PACKAGE_DIR),
        "paths": {
            "vis_npz": str(vis_npz),
            "board_npz": str(board_npz),
            "log": str(log_path),
        },
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


def load_sample_payload(index: int, model_id: str = "completionformer") -> dict:
    model_id = normalize_model_id(model_id)
    if model_id == "cspn":
        return load_cspn_payload(index)
    payload = load_completionformer_payload(index)
    payload["model"] = "completionformer"
    return payload


def run_completionformer_board_sample(index: int) -> dict:
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


def run_cspn_board_sample(index: int) -> dict:
    runner = CSPN_PACKAGE_DIR / "cspn_resnettiny_hw128_w24_step8_board_runner_stagewise_v3_scaleaware.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing CSPN board runner: {runner}")
    input_npz = CSPN_VAL32_INPUT_DIR / f"cspn_val{index:02d}_input.npz"
    if not input_npz.exists():
        raise FileNotFoundError(f"Missing CSPN input NPZ: {input_npz}")
    board = os.environ.get("BOARD", "root@192.168.115.122")
    board_pass = os.environ.get("BOARD_PASS", "root")
    board_pkg = os.environ.get(
        "CSPN_BOARD_PKG",
        "/home/root/workspace/demo_vp_xj/packers/cspn_resnettiny_hw128_w24_step8_stagewise_v3_hostsample",
    )
    remote_input = f"app_inputs/cspn_val{index:02d}_input.npz"
    remote_output = f"app_outputs/cspn_val{index:02d}_board_padded16_clearwr_run1.npz"
    local_output = CSPN_VAL32_BOARD_DIR / f"cspn_val{index:02d}_board_padded16_clearwr_run1.npz"
    prep_cmd = f"mkdir -p '{board_pkg}/app_inputs' '{board_pkg}/app_outputs'"
    prep = subprocess.run(
        [
            "sshpass",
            "-p",
            board_pass,
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            board,
            prep_cmd,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    if prep.returncode != 0:
        return {"returncode": prep.returncode, "elapsed_sec": 0.0, "output_tail": prep.stdout}
    for local_path in [
        runner,
        CSPN_PACKAGE_DIR / "cspn_stagewise_scales_orig_val0_20260709.csv",
        input_npz,
    ]:
        target = f"{board}:{board_pkg}/app_inputs/" if local_path == input_npz else f"{board}:{board_pkg}/"
        scp = subprocess.run(
            [
                "sshpass",
                "-p",
                board_pass,
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                str(local_path),
                target,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        if scp.returncode != 0:
            return {"returncode": scp.returncode, "elapsed_sec": 0.0, "output_tail": scp.stdout}
    cmd = (
        "set -e; "
        f"cd '{board_pkg}'; "
        "python3 cspn_resnettiny_hw128_w24_step8_board_runner_stagewise_v3_scaleaware.py "
        f"'{board_pkg}' --input-npz '{remote_input}' "
        f"--save '{remote_output}' "
        "--scales-csv cspn_stagewise_scales_orig_val0_20260709.csv "
        "--unit-scales --use-scaled-simple --use-scaled-fullsplit --use-padded-depth-head"
    )
    started = time.time()
    proc = subprocess.run(
        [
            "sshpass",
            "-p",
            board_pass,
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            board,
            cmd,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(os.environ.get("CSPN_BOARD_TIMEOUT", "1200")),
    )
    elapsed = time.time() - started
    (CSPN_VAL32_BOARD_DIR / f"cspn_val{index:02d}_app_run.log").write_text(proc.stdout)
    if proc.returncode == 0:
        scp_back = subprocess.run(
            [
                "sshpass",
                "-p",
                board_pass,
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                f"{board}:{board_pkg}/{remote_output}",
                str(local_output),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        if scp_back.returncode == 0 and CSPN_EXPORT_SCRIPT.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(CSPN_EXPORT_SCRIPT),
                    "--sample-index",
                    str(index),
                    "--board-dir",
                    str(CSPN_VAL32_BOARD_DIR),
                    "--out-root",
                    str(CSPN_VIS_ROOT),
                ],
                cwd=str(PORTABLE_DIR),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
    return {
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "output_tail": "\n".join(proc.stdout.splitlines()[-80:]),
    }


def run_board_sample(index: int, model_id: str = "completionformer") -> dict:
    model_id = normalize_model_id(model_id)
    if model_id == "cspn":
        return run_cspn_board_sample(index)
    return run_completionformer_board_sample(index)


def list_samples(model_id: str = "completionformer") -> dict:
    model_id = normalize_model_id(model_id)
    if model_id == "cspn":
        samples = []
        for i in range(32):
            vis_npz = CSPN_VIS_ROOT / f"cspn_sample{i}" / f"cspn_val{i}_padded16_board_pred_outputs.npz"
            samples.append(
                {
                    "index": i,
                    "has_board_output": vis_npz.exists(),
                    "vis_npz": str(vis_npz),
                    "vis_png": "",
                }
            )
        return {
            "model": "cspn",
            "portable_dir": str(CSPN_PACKAGE_DIR),
            "samples": samples,
        }
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
    return {"model": "completionformer", "portable_dir": str(PORTABLE_DIR), "samples": samples}


def list_models() -> dict:
    return {
        "models": [
            {
                "id": "completionformer",
                "name": "CompletionFormer HW128",
                "description": "ckpt00059 RHB conv-only confidence-head deployment",
            },
            {
                "id": "cspn",
                "name": "CSPN ResNetTiny HW128",
                "description": "stagewise scale-aware RHB deployment with padded16 final depth head",
            },
        ]
    }


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        if path == "/" or path.startswith("/static/"):
            if path == "/":
                return str(STATIC_DIR / "index.html")
            return str(APP_DIR / path.lstrip("/"))
        return str(STATIC_DIR / "index.html")

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        model_id = query.get("model", ["completionformer"])[0]
        if parsed.path == "/api/models":
            return self.write_json(list_models())
        if parsed.path == "/api/samples":
            return self.write_json(list_samples(model_id))
        if parsed.path.startswith("/api/sample/"):
            try:
                index = int(parsed.path.rsplit("/", 1)[-1])
                return self.write_json(load_sample_payload(index, model_id))
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
        model_id = query.get("model", ["completionformer"])[0]
        if parsed.path.startswith("/api/run/"):
            try:
                index = int(parsed.path.rsplit("/", 1)[-1])
                run = run_board_sample(index, model_id)
                payload = {"run": run}
                if run["returncode"] == 0:
                    payload["sample"] = load_sample_payload(index, model_id)
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
