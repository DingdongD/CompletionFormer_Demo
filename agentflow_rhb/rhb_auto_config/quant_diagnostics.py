import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class QuantTensorMetrics:
    key: str
    shape: List[int]
    dtype: str
    element_count: int
    channel_axis: int
    min_value: float
    max_value: float
    mean: float
    std: float
    skewness: float
    kurtosis_excess: float
    absmax: float
    abs_p99: float
    abs_p999: float
    top_0p1_abs_energy_ratio: float
    scale_absmax: float
    scale_p999: float
    scale_p99: float
    sat_rate_absmax: float
    sat_rate_p999: float
    sat_rate_p99: float
    qdq_l1_absmax: float
    qdq_l1_p999: float
    qdq_l1_p99: float
    qdq_rmse_absmax: float
    qdq_rmse_p999: float
    qdq_rmse_p99: float
    risk: str
    recommendation: str


@dataclass(frozen=True)
class QuantDiagnosticsReport:
    source_npz: str
    quant_format: str
    tensor_count: int
    metrics: List[QuantTensorMetrics]


def _as_float(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _safe_scale(abs_value: float) -> float:
    return 127.0 / max(float(abs_value), 1.0e-8)


def _moments(arr: np.ndarray) -> Tuple[float, float]:
    flat = _as_float(arr).reshape(-1).astype(np.float64)
    if flat.size == 0:
        return 0.0, 0.0
    std = float(flat.std())
    if std < 1.0e-12:
        return 0.0, -3.0
    centered = flat - float(flat.mean())
    z = centered / std
    skewness = float(np.mean(z ** 3))
    kurtosis_excess = float(np.mean(z ** 4) - 3.0)
    return skewness, kurtosis_excess


def _top_abs_energy_ratio(arr: np.ndarray, top_fraction: float = 0.001) -> float:
    flat = np.abs(_as_float(arr).reshape(-1)).astype(np.float64)
    if flat.size == 0:
        return 0.0
    energy = flat * flat
    total = float(energy.sum())
    if total <= 1.0e-20:
        return 0.0
    k = max(1, int(np.ceil(flat.size * top_fraction)))
    top = np.partition(energy, -k)[-k:]
    return float(top.sum() / total)


def _simulate_qdq(arr: np.ndarray, scale: float) -> Tuple[float, float, float]:
    data = _as_float(arr)
    if data.size == 0 or scale <= 0.0:
        return 0.0, 0.0, 0.0
    q = np.round(data * scale)
    sat = float(np.count_nonzero((q <= -128.0) | (q >= 127.0)) / q.size)
    clipped = np.clip(q, -128.0, 127.0)
    deq = clipped / scale
    diff = deq - data
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff * diff))), sat


def _classify(
    absmax: float,
    abs_p99: float,
    abs_p999: float,
    energy_ratio: float,
    kurtosis_excess: float,
    sat_p999: float,
    l1_p999: float,
) -> Tuple[str, str]:
    risk = "low"
    recommendation = "use absmax or p99.9 scale; validate with board boundary trace"

    if abs_p999 > 0.0 and absmax / max(abs_p999, 1.0e-8) > 4.0:
        risk = "high"
        recommendation = "outlier-dominated tensor; compare absmax, p99.9, and exact split/fusion before board promotion"
    elif abs_p99 > 0.0 and abs_p999 / max(abs_p99, 1.0e-8) > 3.0:
        risk = "medium"
        recommendation = "long-tail tensor; use calibration128 if available and inspect local saturation"

    if energy_ratio > 0.35 or kurtosis_excess > 20.0:
        risk = "high"
        recommendation = "heavy-tail activation; prefer boundary fusion/split or QAT over percentile-only clipping"
    elif energy_ratio > 0.20 or kurtosis_excess > 8.0:
        risk = "medium" if risk == "low" else risk
        recommendation = "moderate outlier risk; record effective board scale and saturation per boundary"

    if sat_p999 > 0.005:
        risk = "high"
        recommendation = "p99.9 scale still saturates; reduce input scale, split high-energy channels, or retrain aligned variant"
    elif l1_p999 > max(abs_p999 * 0.05, 0.02):
        risk = "medium" if risk == "low" else risk
        recommendation = "quantization noise is visible; compare per-channel weight quant and boundary requant placement"

    return risk, recommendation


def analyze_array(key: str, value, channel_axis: int = -1) -> QuantTensorMetrics:
    arr = _as_float(value)
    flat = arr.reshape(-1)
    abs_arr = np.abs(flat)
    absmax = float(abs_arr.max()) if flat.size else 0.0
    abs_p99 = float(np.percentile(abs_arr, 99.0)) if flat.size else 0.0
    abs_p999 = float(np.percentile(abs_arr, 99.9)) if flat.size else 0.0
    scale_absmax = _safe_scale(absmax)
    scale_p999 = _safe_scale(abs_p999)
    scale_p99 = _safe_scale(abs_p99)
    qdq_l1_absmax, qdq_rmse_absmax, sat_absmax = _simulate_qdq(arr, scale_absmax)
    qdq_l1_p999, qdq_rmse_p999, sat_p999 = _simulate_qdq(arr, scale_p999)
    qdq_l1_p99, qdq_rmse_p99, sat_p99 = _simulate_qdq(arr, scale_p99)
    skewness, kurtosis_excess = _moments(arr)
    energy_ratio = _top_abs_energy_ratio(arr)
    risk, recommendation = _classify(
        absmax=absmax,
        abs_p99=abs_p99,
        abs_p999=abs_p999,
        energy_ratio=energy_ratio,
        kurtosis_excess=kurtosis_excess,
        sat_p999=sat_p999,
        l1_p999=qdq_l1_p999,
    )
    return QuantTensorMetrics(
        key=key,
        shape=[int(x) for x in arr.shape],
        dtype=str(np.asarray(value).dtype),
        element_count=int(arr.size),
        channel_axis=int(channel_axis),
        min_value=float(flat.min()) if flat.size else 0.0,
        max_value=float(flat.max()) if flat.size else 0.0,
        mean=float(flat.mean()) if flat.size else 0.0,
        std=float(flat.std()) if flat.size else 0.0,
        skewness=skewness,
        kurtosis_excess=kurtosis_excess,
        absmax=absmax,
        abs_p99=abs_p99,
        abs_p999=abs_p999,
        top_0p1_abs_energy_ratio=energy_ratio,
        scale_absmax=scale_absmax,
        scale_p999=scale_p999,
        scale_p99=scale_p99,
        sat_rate_absmax=sat_absmax,
        sat_rate_p999=sat_p999,
        sat_rate_p99=sat_p99,
        qdq_l1_absmax=qdq_l1_absmax,
        qdq_l1_p999=qdq_l1_p999,
        qdq_l1_p99=qdq_l1_p99,
        qdq_rmse_absmax=qdq_rmse_absmax,
        qdq_rmse_p999=qdq_rmse_p999,
        qdq_rmse_p99=qdq_rmse_p99,
        risk=risk,
        recommendation=recommendation,
    )


def _iter_channel_views(key: str, arr: np.ndarray, axis: int, max_channels: int) -> List[Tuple[str, np.ndarray]]:
    if axis < 0:
        axis = arr.ndim + axis
    if axis < 0 or axis >= arr.ndim:
        return []
    count = min(int(arr.shape[axis]), int(max_channels))
    views = []
    for idx in range(count):
        view = np.take(arr, idx, axis=axis)
        views.append((f"{key}/ch{idx:04d}", view))
    return views


def analyze_npz(
    npz_path: Path,
    keys: Optional[Sequence[str]] = None,
    channel_axis: Optional[int] = None,
    max_channel_items: int = 512,
) -> QuantDiagnosticsReport:
    data = np.load(npz_path)
    selected = list(keys) if keys else sorted(data.files)
    metrics: List[QuantTensorMetrics] = []
    for key in selected:
        if key not in data.files:
            continue
        arr = data[key]
        if arr.dtype.kind not in "fiu":
            continue
        metrics.append(analyze_array(key, arr, channel_axis=channel_axis if channel_axis is not None else -1))
        if channel_axis is not None and arr.ndim > 1:
            for ch_key, ch_arr in _iter_channel_views(key, arr, channel_axis, max_channel_items):
                metrics.append(analyze_array(ch_key, ch_arr, channel_axis=-1))
    return QuantDiagnosticsReport(
        source_npz=str(npz_path),
        quant_format="int8_symmetric_activation_per_tensor; optional per-channel diagnostics only",
        tensor_count=len(metrics),
        metrics=metrics,
    )


def save_quant_diagnostics(report: QuantDiagnosticsReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def render_quant_diagnostics(report: QuantDiagnosticsReport, limit: int = 120) -> str:
    lines = [
        "# Quant Diagnostics",
        "",
        f"- source: `{report.source_npz}`",
        f"- quant_format: `{report.quant_format}`",
        f"- tensor_count: {report.tensor_count}",
        "",
        "risk\tkey\tshape\tabsmax\tabs_p999\tkurtosis\ttop0.1_energy\tsat_p999\tqdq_l1_p999\tscale_absmax\tscale_p999\trecommendation",
    ]
    ordered = sorted(report.metrics, key=lambda item: ({"high": 0, "medium": 1, "low": 2}.get(item.risk, 3), -item.qdq_l1_p999))
    for item in ordered[:limit]:
        shape = "x".join(str(x) for x in item.shape)
        lines.append(
            f"{item.risk}\t{item.key}\t{shape}\t{item.absmax:.6g}\t{item.abs_p999:.6g}\t"
            f"{item.kurtosis_excess:.6g}\t{item.top_0p1_abs_energy_ratio:.6g}\t"
            f"{item.sat_rate_p999:.6g}\t{item.qdq_l1_p999:.6g}\t"
            f"{item.scale_absmax:.6g}\t{item.scale_p999:.6g}\t{item.recommendation}"
        )
    return "\n".join(lines)
