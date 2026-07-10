#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "data" / "nyu_val32_source_128x128.npz"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "cspn_unified_input" / "inputs"))
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(source)
    rgb_norm = z["rgb"].astype(np.float32)
    dep = z["dep"].astype(np.float32)
    gt = z["gt"].astype(np.float32)
    rgb01 = np.clip(rgb_norm * STD + MEAN, 0.0, 1.0)
    rgbd = np.concatenate([rgb01, dep], axis=1).astype(np.float32)
    input_scale = float(args.input_scale) if args.input_scale is not None else 127.0 / max(float(np.max(np.abs(rgbd))), 1.0e-6)

    indices = [args.sample_index] if args.sample_index is not None else list(range(rgbd.shape[0]))
    for idx in indices:
        input0_i8 = np.clip(np.rint(rgbd[idx : idx + 1] * input_scale), -128, 127).astype(np.int8)
        out = out_dir / f"cspn_val{idx:02d}_input.npz"
        np.savez_compressed(
            out,
            input0_i8=input0_i8,
            rgbd=rgbd[idx : idx + 1],
            rgb=rgb01[idx : idx + 1],
            sparse=dep[idx : idx + 1],
            gt=gt[idx : idx + 1],
            input_scale=np.array(input_scale, dtype=np.float32),
            source_npz=np.array(str(source)),
            source_index=np.array(idx, dtype=np.int32),
            source_kind=np.array("completionformer_nyu_val32_source_128x128"),
        )
        out.with_suffix(".json").write_text(
            json.dumps(
                {
                    "source_npz": str(source),
                    "source_index": idx,
                    "input_scale": input_scale,
                    "rgb_display": "denormalized CompletionFormer RGB, clipped to [0,1]",
                    "sparse_depth": "CompletionFormer dep tensor",
                    "gt": "CompletionFormer gt tensor",
                },
                indent=2,
            )
        )
        print(f"sample={idx} saved={out}")


if __name__ == "__main__":
    main()
