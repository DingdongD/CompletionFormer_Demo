#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


EXPECTED_CONTRACT = "board_pred is the final displayed/runtime output; ref_pred is comparison-only"
EXPECTED_ERROR_MODE = "board_vs_ref"
REQUIRED_IMAGES = ("rgb", "sparse_depth", "gt", "ref_pred", "board_pred", "abs_error")


def audit(root: Path) -> int:
    manifest_path = root / "docs" / "data" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples", [])
    counts = Counter(sample.get("model", "completionformer") for sample in samples)
    failures: list[str] = []

    for sample in samples:
        sample_id = sample.get("id", str(sample.get("index")))
        base = root / "docs" / sample["base"]
        meta_path = base / "meta.json"
        if not meta_path.exists():
            failures.append(f"{sample_id}: missing meta.json")
            continue
        meta = json.loads(meta_path.read_text())
        images = meta.get("images", {})
        metrics = meta.get("metrics", {})

        for key in REQUIRED_IMAGES:
            image_name = images.get(key)
            if not image_name or not (base / image_name).exists():
                failures.append(f"{sample_id}: missing image {key}")
        point_cloud = meta.get("point_cloud")
        if not point_cloud or not (base / point_cloud).exists():
            failures.append(f"{sample_id}: missing board_pred point cloud")
        if metrics.get("output_contract") != EXPECTED_CONTRACT:
            failures.append(f"{sample_id}: missing board output contract")
        if metrics.get("error_image_mode") != EXPECTED_ERROR_MODE:
            failures.append(f"{sample_id}: missing board-vs-ref error mode")
        if "board_ref_abs_mean" not in metrics or "board_ref_rmse" not in metrics:
            failures.append(f"{sample_id}: missing board-ref metrics")
        if "board_gt_l1" not in metrics or "board_gt_rmse" not in metrics:
            failures.append(f"{sample_id}: missing board-gt metrics")

    print(f"samples={len(samples)} counts={dict(counts)}")
    if failures:
        print("FAIL")
        for failure in failures:
            print(failure)
        return 1
    print("PASS board output contract")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="portable_runtime root",
    )
    args = parser.parse_args()
    return audit(args.root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
