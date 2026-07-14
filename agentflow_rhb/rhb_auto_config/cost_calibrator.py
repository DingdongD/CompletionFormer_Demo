import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class CostSample:
    path: str
    model: str
    status: str
    latency_ms: float
    counters: Dict[str, int]


@dataclass(frozen=True)
class CostCalibration:
    result_root: str
    samples: List[CostSample]
    pass_count: int
    fail_count: int
    avg_pass_latency_ms: float
    max_pass_latency_ms: float


def _extract_sample(path: Path) -> CostSample:
    data = json.loads(path.read_text(encoding="utf-8"))
    model = str(data.get("model") or data.get("packer_dir") or path.stem)
    parsed = {}
    if data.get("board"):
        parsed = data["board"].get("parsed", {})
    elif data.get("parsed"):
        parsed = data.get("parsed", {})
    status = str(parsed.get("status", "unknown"))
    latency = parsed.get("latency", {}) or {}
    latency_values = [float(v) for v in latency.values()] if isinstance(latency, dict) else []
    latency_ms = sum(latency_values) if latency_values else 0.0
    counters = {str(k): int(v) for k, v in (parsed.get("counters", {}) or {}).items()}
    return CostSample(
        path=str(path),
        model=model,
        status=status,
        latency_ms=round(latency_ms, 4),
        counters=counters,
    )


def calibrate_costs(result_root: Path) -> CostCalibration:
    samples: List[CostSample] = []
    for path in sorted(result_root.rglob("*.json")):
        try:
            sample = _extract_sample(path)
        except Exception:
            continue
        if sample.status == "unknown" and not sample.counters:
            continue
        samples.append(sample)
    pass_samples = [item for item in samples if item.status == "pass"]
    pass_latencies = [item.latency_ms for item in pass_samples if item.latency_ms > 0]
    avg = sum(pass_latencies) / len(pass_latencies) if pass_latencies else 0.0
    max_latency = max(pass_latencies) if pass_latencies else 0.0
    return CostCalibration(
        result_root=str(result_root),
        samples=samples,
        pass_count=len(pass_samples),
        fail_count=len(samples) - len(pass_samples),
        avg_pass_latency_ms=round(avg, 4),
        max_pass_latency_ms=round(max_latency, 4),
    )


def save_cost_calibration(calibration: CostCalibration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")


def render_cost_calibration(calibration: CostCalibration) -> str:
    lines = [
        "# Cost Calibration",
        "",
        f"- result root: `{calibration.result_root}`",
        f"- samples: {len(calibration.samples)}",
        f"- pass: {calibration.pass_count}",
        f"- fail/unknown: {calibration.fail_count}",
        f"- avg pass latency: {calibration.avg_pass_latency_ms} ms",
        f"- max pass latency: {calibration.max_pass_latency_ms} ms",
        "",
        "## Samples",
        "",
        "status\tlatency_ms\tmodel\tpath",
    ]
    for sample in calibration.samples[:200]:
        lines.append(f"{sample.status}\t{sample.latency_ms}\t{sample.model}\t{sample.path}")
    return "\n".join(lines)
