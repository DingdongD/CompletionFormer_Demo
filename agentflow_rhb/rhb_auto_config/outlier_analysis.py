import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class OutlierMetrics:
    key: str
    l1: float
    rmse: float
    corr: float
    ref_min: float
    ref_max: float
    ref_p99: float
    ref_p999: float
    ref_absmax: float
    ref_abs_p99: float
    ref_abs_p999: float
    board_min: float
    board_max: float
    board_p99: float
    board_p999: float
    board_absmax: float
    board_abs_p99: float
    board_abs_p999: float
    max_compression: float
    p999_compression: float
    top_0p1_abs_energy_ratio: float
    suggested_scale_max: float
    suggested_scale_p999: float
    int8_saturation_rate: float
    risk: str
    likely_outlier_driven: bool
    recommendation: str


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = _as_float_array(a).reshape(-1)
    b = _as_float_array(b).reshape(-1)
    if a.size == 0 or float(np.std(a)) < 1.0e-8 or float(np.std(b)) < 1.0e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_ratio(num: float, den: float) -> float:
    if abs(float(den)) < 1.0e-12:
        return float("inf") if abs(float(num)) > 1.0e-12 else 1.0
    return float(num) / float(den)


def _abs_percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(np.abs(_as_float_array(x)), q))


def _signed_percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(_as_float_array(x), q))


def _top_abs_energy_ratio(x: np.ndarray, top_fraction: float = 0.001) -> float:
    flat = np.abs(_as_float_array(x).reshape(-1)).astype(np.float64)
    if flat.size == 0:
        return 0.0
    energy = flat * flat
    total = float(energy.sum())
    if total <= 1.0e-20:
        return 0.0
    k = max(1, int(np.ceil(flat.size * top_fraction)))
    top = np.partition(energy, -k)[-k:]
    return float(top.sum() / total)


def _saturation_rate(x: np.ndarray) -> float:
    arr = _as_float_array(x)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero((arr <= -128.0) | (arr >= 127.0)) / arr.size)


def _risk_and_recommendation(
    l1: float,
    corr: float,
    max_compression: float,
    p999_compression: float,
    energy_ratio: float,
    saturation_rate: float,
) -> Tuple[str, bool, str]:
    likely = False
    risk = "low"
    recommendation = "keep current scale contract"

    if max_compression < 0.35 or p999_compression < 0.45:
        likely = True
        risk = "high"
        recommendation = "widen activation scale coverage or add outlier-aware boundary/QAT"
    elif max_compression < 0.6 or p999_compression < 0.7:
        likely = True
        risk = "medium"
        recommendation = "evaluate p99.9/max scale candidates and layerwise board-vs-ref"

    if energy_ratio > 0.35 and (max_compression < 0.7 or p999_compression < 0.8):
        likely = True
        risk = "high"
        recommendation = "use outlier-aware calibration; test max-scale and percentile-scale tradeoff"

    if saturation_rate > 0.001:
        likely = True
        risk = "high"
        recommendation = "reduce input scale or split/fuse boundary to avoid int8 saturation"

    if corr < 0.5 and risk != "high":
        risk = "medium"
        recommendation = "localize first divergent boundary before changing model structure"

    if l1 > 1.0 and corr < 0.3:
        risk = "high"
        recommendation = "treat boundary as deployment-contract failure; retrain/QAT with exact split contract"

    return risk, likely, recommendation


def analyze_pair(key: str, board, ref) -> OutlierMetrics:
    board_arr = _as_float_array(board)
    ref_arr = _as_float_array(ref)
    diff = board_arr - ref_arr
    l1 = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    corr = _corrcoef(board_arr, ref_arr)

    ref_absmax = float(np.max(np.abs(ref_arr))) if ref_arr.size else 0.0
    board_absmax = float(np.max(np.abs(board_arr))) if board_arr.size else 0.0
    ref_abs_p999 = _abs_percentile(ref_arr, 99.9)
    board_abs_p999 = _abs_percentile(board_arr, 99.9)
    max_compression = _safe_ratio(board_absmax, ref_absmax)
    p999_compression = _safe_ratio(board_abs_p999, ref_abs_p999)
    energy_ratio = _top_abs_energy_ratio(ref_arr)
    suggested_scale_max = 127.0 / max(ref_absmax, 1.0e-6)
    suggested_scale_p999 = 127.0 / max(ref_abs_p999, 1.0e-6)
    saturation_rate = _saturation_rate(board_arr)

    risk, likely, recommendation = _risk_and_recommendation(
        l1=l1,
        corr=corr,
        max_compression=max_compression,
        p999_compression=p999_compression,
        energy_ratio=energy_ratio,
        saturation_rate=saturation_rate,
    )

    return OutlierMetrics(
        key=key,
        l1=l1,
        rmse=rmse,
        corr=corr,
        ref_min=float(ref_arr.min()) if ref_arr.size else 0.0,
        ref_max=float(ref_arr.max()) if ref_arr.size else 0.0,
        ref_p99=_signed_percentile(ref_arr, 99.0),
        ref_p999=_signed_percentile(ref_arr, 99.9),
        ref_absmax=ref_absmax,
        ref_abs_p99=_abs_percentile(ref_arr, 99.0),
        ref_abs_p999=ref_abs_p999,
        board_min=float(board_arr.min()) if board_arr.size else 0.0,
        board_max=float(board_arr.max()) if board_arr.size else 0.0,
        board_p99=_signed_percentile(board_arr, 99.0),
        board_p999=_signed_percentile(board_arr, 99.9),
        board_absmax=board_absmax,
        board_abs_p99=_abs_percentile(board_arr, 99.0),
        board_abs_p999=board_abs_p999,
        max_compression=max_compression,
        p999_compression=p999_compression,
        top_0p1_abs_energy_ratio=energy_ratio,
        suggested_scale_max=suggested_scale_max,
        suggested_scale_p999=suggested_scale_p999,
        int8_saturation_rate=saturation_rate,
        risk=risk,
        likely_outlier_driven=likely,
        recommendation=recommendation,
    )


def analyze_npz_pairs(
    ref_npz: Path,
    board_npz: Path,
    keys: Optional[Sequence[str]] = None,
) -> List[OutlierMetrics]:
    ref = np.load(ref_npz)
    board = np.load(board_npz)
    selected = list(keys) if keys else sorted(set(ref.files).intersection(board.files))
    metrics = []
    for key in selected:
        if key not in ref.files or key not in board.files:
            continue
        ref_arr = ref[key]
        board_arr = board[key]
        if ref_arr.shape != board_arr.shape:
            continue
        if ref_arr.dtype.kind not in "fiu" or board_arr.dtype.kind not in "fiu":
            continue
        metrics.append(analyze_pair(key, board_arr, ref_arr))
    return metrics


def load_boundary_csv(path: Path) -> List[OutlierMetrics]:
    """Load a boundary CSV with ranges and correlation, then derive outlier risk.

    This mode cannot recover percentiles because arrays are not available. It is
    intended for historical reports and marks unavailable percentile fields as
    NaN while still classifying max-range compression.
    """

    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            board_absmax = max(abs(float(row["board_min"])), abs(float(row["board_max"])))
            ref_absmax = max(abs(float(row["ref_min"])), abs(float(row["ref_max"])))
            max_compression = _safe_ratio(board_absmax, ref_absmax)
            l1 = float(row["l1"])
            rmse = float(row["rmse"])
            corr = float(row["corr"])
            risk, likely, recommendation = _risk_and_recommendation(
                l1=l1,
                corr=corr,
                max_compression=max_compression,
                p999_compression=float("nan"),
                energy_ratio=float("nan"),
                saturation_rate=0.0,
            )
            rows.append(
                OutlierMetrics(
                    key=f"{row.get('mode', 'unknown')}:{row['key']}",
                    l1=l1,
                    rmse=rmse,
                    corr=corr,
                    ref_min=float(row["ref_min"]),
                    ref_max=float(row["ref_max"]),
                    ref_p99=float("nan"),
                    ref_p999=float("nan"),
                    ref_absmax=ref_absmax,
                    ref_abs_p99=float("nan"),
                    ref_abs_p999=float("nan"),
                    board_min=float(row["board_min"]),
                    board_max=float(row["board_max"]),
                    board_p99=float("nan"),
                    board_p999=float("nan"),
                    board_absmax=board_absmax,
                    board_abs_p99=float("nan"),
                    board_abs_p999=float("nan"),
                    max_compression=max_compression,
                    p999_compression=float("nan"),
                    top_0p1_abs_energy_ratio=float("nan"),
                    suggested_scale_max=127.0 / max(ref_absmax, 1.0e-6),
                    suggested_scale_p999=float("nan"),
                    int8_saturation_rate=0.0,
                    risk=risk,
                    likely_outlier_driven=likely,
                    recommendation=recommendation,
                )
            )
    return rows


def save_outlier_json(metrics: Iterable[OutlierMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in metrics], indent=2), encoding="utf-8")


def render_outlier_report(metrics: Iterable[OutlierMetrics], limit: int = 80) -> str:
    items = list(metrics)
    lines = [
        "key\trisk\toutlier_driven\tl1\trmse\tcorr\tmax_compression\tp999_compression\t"
        "ref_absmax\tboard_absmax\tref_abs_p999\tboard_abs_p999\ttop0p1_energy\t"
        "sat_rate\tscale_max\tscale_p999\trecommendation"
    ]
    for item in items[:limit]:
        lines.append(
            f"{item.key}\t{item.risk}\t{item.likely_outlier_driven}\t"
            f"{item.l1:.6g}\t{item.rmse:.6g}\t{item.corr:.6g}\t"
            f"{item.max_compression:.6g}\t{item.p999_compression:.6g}\t"
            f"{item.ref_absmax:.6g}\t{item.board_absmax:.6g}\t"
            f"{item.ref_abs_p999:.6g}\t{item.board_abs_p999:.6g}\t"
            f"{item.top_0p1_abs_energy_ratio:.6g}\t{item.int8_saturation_rate:.6g}\t"
            f"{item.suggested_scale_max:.6g}\t{item.suggested_scale_p999:.6g}\t"
            f"{item.recommendation}"
        )
    return "\n".join(lines)
