import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FailureFinding:
    severity: str
    category: str
    likely_cause: str
    evidence: str
    recommended_action: str


@dataclass(frozen=True)
class FailureLocalization:
    result_path: str
    status: str
    model: str
    findings: List[FailureFinding]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_from_result(data: Dict) -> str:
    if data.get("board"):
        return str(data["board"].get("parsed", {}).get("status", "unknown"))
    if data.get("parsed"):
        return str(data["parsed"].get("status", "unknown"))
    if data.get("compile"):
        compile_status = data["compile"].get("compile", {}).get("status", "")
        cmodel_status = data["compile"].get("cmodel", {}).get("status", "")
        if compile_status == "fail":
            return "compile_fail"
        if cmodel_status == "fail":
            return "cmodel_fail"
    return "unknown"


def localize_failure(result_json: Path, region_plan_json: Optional[Path] = None) -> FailureLocalization:
    data = _load_json(result_json)
    status = _status_from_result(data)
    model = str(data.get("model") or data.get("packer_dir") or result_json.stem)
    findings: List[FailureFinding] = []

    board_errors: List[str] = []
    counters: Dict[str, int] = {}
    if data.get("board"):
        parsed = data["board"].get("parsed", {})
        board_errors = [str(item) for item in parsed.get("errors", [])]
        counters = {str(k): int(v) for k, v in parsed.get("counters", {}).items()}
    elif data.get("parsed"):
        parsed = data.get("parsed", {})
        board_errors = [str(item) for item in parsed.get("errors", [])]
        counters = {str(k): int(v) for k, v in parsed.get("counters", {}).items()}

    errors_text = "\n".join(board_errors)
    if status == "pass":
        findings.append(
            FailureFinding(
                severity="info",
                category="pass",
                likely_cause="no failure detected",
                evidence=str(result_json),
                recommended_action="promote exact shape/pattern to reviewed allow rule if numerical validation is sufficient",
            )
        )
    elif status == "fail_timeout":
        opt_cnt = counters.get("sta_npu_opt_cnt", -1)
        if opt_cnt <= 0:
            cause = "runtime did not launch or status/done flag was stale"
            action = "force rram_only=false, clear wr_done before run, verify DMA buffer sync, then retry smallest region"
        else:
            cause = "accelerator started but did not complete; likely unsupported shape/op sequence or output writeback stall"
            action = "binary split the RHB region; forbid the first failing half or rewrite risky op"
        findings.append(
            FailureFinding(
                severity="high",
                category="board_timeout",
                likely_cause=cause,
                evidence=errors_text or f"counters={counters}",
                recommended_action=action,
            )
        )
    elif status == "fail_accuracy":
        findings.append(
            FailureFinding(
                severity="high",
                category="board_accuracy",
                likely_cause="board completed but output differs; likely mis-lowered activation, quantization scale mismatch, or stale output buffer",
                evidence=errors_text or str(result_json),
                recommended_action="run first-output compare, split around activation/glue, verify dequant/requant scale at Host/RHB boundary",
            )
        )
    elif status == "fail_runtime":
        if "must have exactly one output tensor" in errors_text:
            cause = "board runtime only accepts one output tensor per submodel"
            action = "split multi-output RHB region into one submodel per output, or keep fan-out boundary on Host"
        else:
            cause = "board runtime failed before numerical comparison"
            action = "inspect ac_driver load/run error, then split at runtime boundary or add a custom runner"
        findings.append(
            FailureFinding(
                severity="high",
                category="board_runtime",
                likely_cause=cause,
                evidence=errors_text or str(result_json),
                recommended_action=action,
            )
        )
    elif status in {"compile_fail", "cmodel_fail"}:
        findings.append(
            FailureFinding(
                severity="medium",
                category=status,
                likely_cause="unsupported compiler pattern or invalid layout/shape",
                evidence=str(result_json),
                recommended_action="apply rewrite rules before board testing; keep unsupported op on Host",
            )
        )
    else:
        findings.append(
            FailureFinding(
                severity="medium",
                category="unknown",
                likely_cause="insufficient structured evidence",
                evidence=str(result_json),
                recommended_action="collect compile/cmodel/board logs and rerun localizer",
            )
        )

    if region_plan_json and region_plan_json.exists():
        plan = _load_json(region_plan_json)
        risky_regions = []
        for region in plan.get("regions", []):
            if region.get("target") != "RHB":
                continue
            if region.get("rewrite_suggestions"):
                risky_regions.append(region.get("region_id"))
        if risky_regions:
            findings.append(
                FailureFinding(
                    severity="medium",
                    category="region_risk",
                    likely_cause="RHB candidate contains rewrite/probe-risk pattern",
                    evidence=", ".join(str(x) for x in risky_regions),
                    recommended_action="split region at the suggested rewrite node or force that node to Host",
                )
            )

    return FailureLocalization(
        result_path=str(result_json),
        status=status,
        model=model,
        findings=findings,
    )


def save_failure_localization(localization: FailureLocalization, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(localization), indent=2), encoding="utf-8")


def render_failure_localization(localization: FailureLocalization) -> str:
    lines = [
        "# Failure Localization",
        "",
        f"- result: `{localization.result_path}`",
        f"- model: `{localization.model}`",
        f"- status: {localization.status}",
        "",
    ]
    for idx, finding in enumerate(localization.findings, 1):
        lines.extend(
            [
                f"## Finding {idx}: {finding.category}",
                "",
                f"- severity: {finding.severity}",
                f"- likely cause: {finding.likely_cause}",
                f"- evidence: {finding.evidence}",
                f"- recommended action: {finding.recommended_action}",
                "",
            ]
        )
    return "\n".join(lines)
