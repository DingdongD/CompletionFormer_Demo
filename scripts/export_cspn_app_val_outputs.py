#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path("/root/demo")
TRAINING_ROOT = DEMO_ROOT / "artifacts" / "rhb_auto_config_framework" / "training" / "cspn_aligned_hw"
sys.path.insert(0, str(TRAINING_ROOT))

from cspn_resnet_tiny_aligned_hw import ResNetTinyAlignedCSPNHW  # noqa: E402


def load_model(ckpt_path: Path):
    model = ResNetTinyAlignedCSPNHW(
        width=24,
        cspn_step=8,
        cspn_norm_type="8sum",
        downsample_variant="sample_1x1",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_one(model, input_path: Path, board_path: Path, out_path: Path) -> dict:
    inp = np.load(input_path)
    board = np.load(board_path)
    rgbd = inp["rgbd"].astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(rgbd)
        ref_pred = model(x)[0, 0].cpu().numpy()
        ref_raw, ref_guidance = model.forward_heads(x)
        ref_raw = ref_raw[0, 0].cpu().numpy()
        ref_guidance = ref_guidance[0].cpu().numpy()
        board_raw_t = torch.from_numpy(board["depth_float"].astype(np.float32))
        board_guidance_t = torch.from_numpy(board["guidance_float"].astype(np.float32))
        board_pred = model.cspn(board_guidance_t, board_raw_t, x[:, 3:4])[0, 0].cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ref_pred=ref_pred.astype(np.float32),
        ref_raw=ref_raw.astype(np.float32),
        ref_guidance=ref_guidance.astype(np.float32),
        board_raw=board["depth_float"].astype(np.float32),
        board_guidance=board["guidance_float"].astype(np.float32),
        board_pred=board_pred.astype(np.float32),
        gt=inp["gt"][0, 0].astype(np.float32),
        rgb=inp["rgb"][0].astype(np.float32),
        sparse=inp["sparse"][0, 0].astype(np.float32),
        input_npz=np.array(str(input_path)),
        board_npz=np.array(str(board_path)),
    )
    diff = board_pred - ref_pred
    return {
        "out": str(out_path),
        "l1": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "outputs" / "cspn_unified_input" / "inputs"),
    )
    parser.add_argument(
        "--board-dir",
        default=str(ROOT / "outputs" / "cspn_unified_input" / "board_outputs"),
    )
    parser.add_argument(
        "--ckpt",
        default="/root/demo/artifacts/cspn_aligned_hw_ckpts/nyu_cspn_resnettiny_hw128_w24_step8_20260708_064340/best_model.pt",
    )
    parser.add_argument("--out-root", default=str(ROOT / "outputs"))
    parser.add_argument("--sample-index", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    board_dir = Path(args.board_dir)
    out_root = Path(args.out_root)
    model = load_model(Path(args.ckpt))

    indices = [args.sample_index] if args.sample_index is not None else list(range(32))
    for idx in indices:
        input_path = input_dir / f"cspn_val{idx:02d}_input.npz"
        board_path = board_dir / f"cspn_val{idx:02d}_board_padded16_clearwr_run1.npz"
        if not board_path.exists():
            board_path = board_dir / f"cspn_val{idx:02d}_board_padded16_app.npz"
        out_path = out_root / f"cspn_sample{idx}" / f"cspn_val{idx}_padded16_board_pred_outputs.npz"
        result = export_one(model, input_path, board_path, out_path)
        print(f"sample={idx} l1={result['l1']:.6f} rmse={result['rmse']:.6f} out={result['out']}")


if __name__ == "__main__":
    main()
