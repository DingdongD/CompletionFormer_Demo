import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TensorCompare:
    name: str
    l1: float
    rmse: float
    corr: float
    board_min: float
    board_max: float
    ref_min: float
    ref_max: float
    board_mean: float
    ref_mean: float
    board_std: float
    ref_std: float
    scale_board_div_to_ref: float


@dataclass(frozen=True)
class LayerwiseDecision:
    name: str
    decision: str
    reason: str


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32).reshape(-1)
    rhs = np.asarray(b, dtype=np.float32).reshape(-1)
    if lhs.size == 0 or float(lhs.std()) < 1.0e-8 or float(rhs.std()) < 1.0e-8:
        return float("nan")
    return float(np.corrcoef(lhs, rhs)[0, 1])


def compare_tensors(name: str, board: np.ndarray, ref: np.ndarray) -> TensorCompare:
    board = np.asarray(board, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    diff = board - ref
    dot = float(board.reshape(-1) @ ref.reshape(-1))
    scale = float((board.reshape(-1) @ board.reshape(-1)) / dot) if abs(dot) > 1.0e-12 else float("nan")
    return TensorCompare(
        name=name,
        l1=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff * diff))),
        corr=corrcoef(board, ref),
        board_min=float(board.min()),
        board_max=float(board.max()),
        ref_min=float(ref.min()),
        ref_max=float(ref.max()),
        board_mean=float(board.mean()),
        ref_mean=float(ref.mean()),
        board_std=float(board.std()),
        ref_std=float(ref.std()),
        scale_board_div_to_ref=scale,
    )


def decide_layer_allocation(
    item: TensorCompare,
    corr_pass: float = 0.995,
    corr_probe: float = 0.98,
    relative_l1_probe: float = 0.05,
) -> LayerwiseDecision:
    ref_span = max(abs(item.ref_max - item.ref_min), 1.0e-6)
    relative_l1 = item.l1 / ref_span
    if item.corr >= corr_pass and relative_l1 <= relative_l1_probe:
        return LayerwiseDecision(item.name, "keep_on_rhb", "board tensor matches reference closely")
    if item.corr >= corr_probe:
        return LayerwiseDecision(
            item.name,
            "probe_scale_or_channel_correction",
            "spatial structure is preserved but amplitude/offset differs",
        )
    return LayerwiseDecision(
        item.name,
        "split_before_or_host",
        "layer output diverges structurally from reference",
    )


def fit_scale_bias(src: np.ndarray, dst: np.ndarray, per_channel: bool = False):
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    out = np.empty_like(src)
    params = []
    channels = src.shape[1] if per_channel else 1
    for idx in range(channels):
        s = src[:, idx : idx + 1] if per_channel else src
        d = dst[:, idx : idx + 1] if per_channel else dst
        x = s.reshape(-1).astype(np.float64)
        y = d.reshape(-1).astype(np.float64)
        var = float(np.var(x))
        if var < 1.0e-12:
            a = 1.0
            b = float(y.mean() - x.mean())
        else:
            a = float(np.cov(x, y, bias=True)[0, 1] / var)
            b = float(y.mean() - a * x.mean())
        if per_channel:
            out[:, idx : idx + 1] = src[:, idx : idx + 1] * a + b
        else:
            out = src * a + b
        params.append((a, b))
    return out, np.asarray(params, dtype=np.float32)


def save_tensor_compares(items: Iterable[TensorCompare], path: Path) -> None:
    rows: List[dict] = [asdict(item) for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_compares(items: Sequence[TensorCompare]) -> Dict[str, float]:
    if not items:
        return {}
    return {
        "l1_mean": float(np.mean([item.l1 for item in items])),
        "l1_max": float(np.max([item.l1 for item in items])),
        "rmse_mean": float(np.mean([item.rmse for item in items])),
        "rmse_max": float(np.max([item.rmse for item in items])),
        "corr_mean": float(np.nanmean([item.corr for item in items])),
        "corr_min": float(np.nanmin([item.corr for item in items])),
    }


def save_compare_summary(groups: Mapping[str, Sequence[TensorCompare]], path: Path) -> None:
    rows = []
    for name, items in groups.items():
        row = {"name": name}
        row.update(summarize_compares(list(items)))
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["name", "l1_mean", "l1_max", "rmse_mean", "rmse_max", "corr_mean", "corr_min"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def oracle_ablation(
    board_tensors: Mapping[str, np.ndarray],
    ref_tensors: Mapping[str, np.ndarray],
    output_name: str,
    run_fn: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    variants: Mapping[str, Sequence[str]],
) -> Dict[str, TensorCompare]:
    """Evaluate which tensor boundaries dominate final-output error.

    `run_fn` should consume a tensor dictionary and return the final output.
    Each variant names the board tensors that should be replaced by reference
    tensors before calling `run_fn`.
    """

    results: Dict[str, TensorCompare] = {}
    ref_output = np.asarray(ref_tensors[output_name], dtype=np.float32)
    for variant, replace_names in variants.items():
        payload = {key: np.asarray(value).copy() for key, value in board_tensors.items()}
        for name in replace_names:
            payload[name] = np.asarray(ref_tensors[name], dtype=np.float32)
        out = run_fn(payload)
        results[variant] = compare_tensors(variant, out, ref_output)
    return results
