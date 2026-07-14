import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class HealthItem:
    source: str
    kind: str
    name: str
    package_dir: str = ""
    validation_json: str = ""
    source_onnx: str = ""
    model_name: str = ""
    layout: str = ""
    compile_status: str = ""
    cmodel_status: str = ""
    pack_status: str = ""
    board_status: str = ""
    board_all_same: str = ""
    latency_ms: Optional[float] = None
    diagnosis: str = ""
    recommended_action: str = ""

    @property
    def terminal_status(self) -> str:
        values = [self.compile_status, self.cmodel_status, self.pack_status, self.board_status]
        joined = " ".join(str(v).lower() for v in values if v)
        board = str(self.board_status).lower()
        same = str(self.board_all_same).lower()
        if any(x in joined for x in ["fail", "error", "timeout"]):
            return "fail"
        if self.board_all_same and same not in {"true", "1", "yes", "pass", "-", "none"}:
            return "fail"
        if board == "pass" and (not self.board_all_same or same in {"true", "1", "yes", "pass"}):
            return "pass"
        if self.compile_status == "pass" and self.cmodel_status in {"pass", "pass_with_warnings"} and not self.board_status:
            return "pass"
        if any(x in joined for x in ["warn", "warning"]):
            return "warn"
        if any(v for v in values) or self.board_all_same:
            return "pass"
        return "unknown"


@dataclass
class HealthReport:
    roots: List[str]
    items: List[HealthItem]
    summary: Dict[str, Any]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _latency_from_dict(data: Dict[str, Any]) -> Optional[float]:
    for key in ("latency_ms", "latency_total_ms", "total_latency_ms"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    latency = data.get("latency")
    if isinstance(latency, dict):
        for key in ("total_ms", "decoder_system_128x128_ckpt_tracked_total"):
            value = latency.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _result_status(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    return str(value)


def _item_from_deploy_loop(path: Path, data: Dict[str, Any]) -> HealthItem:
    compile_data = data.get("compile") or {}
    pack_data = data.get("pack") or {}
    board_data = data.get("board") or {}
    board_parsed = board_data.get("parsed") if isinstance(board_data, dict) else {}
    board_parsed = board_parsed or {}
    compile_compile = compile_data.get("compile") if isinstance(compile_data, dict) else {}
    compile_cmodel = compile_data.get("cmodel") if isinstance(compile_data, dict) else {}
    compile_compile = compile_compile or {}
    compile_cmodel = compile_cmodel or {}
    pack_status = ""
    if isinstance(pack_data, dict):
        rc = pack_data.get("returncode")
        if rc is not None:
            pack_status = "pass" if rc == 0 else f"fail_rc_{rc}"
    errors = []
    if isinstance(board_parsed, dict):
        errors.extend(str(x) for x in board_parsed.get("errors", [])[:2])
    return HealthItem(
        source=str(path),
        kind="deploy_loop",
        name=str(data.get("model") or path.stem.replace("deploy_loop_", "")),
        compile_status=_result_status(compile_compile.get("status")),
        cmodel_status=_result_status(compile_cmodel.get("status")),
        pack_status=pack_status,
        board_status=_result_status(board_parsed.get("status")),
        board_all_same=_result_status(board_parsed.get("all_same")),
        latency_ms=_latency_from_dict(board_parsed),
        diagnosis="; ".join(errors),
    )


def _item_from_package_validation(path: Path, item: Dict[str, Any], package_dir: str = "") -> HealthItem:
    return HealthItem(
        source=str(path),
        kind="package_item",
        name=str(item.get("region_id") or item.get("model_name") or "unknown"),
        package_dir=package_dir,
        validation_json=str(path),
        source_onnx=str(item.get("source_onnx") or ""),
        model_name=str(item.get("model_name") or ""),
        layout=str(item.get("layout") or ""),
        compile_status=_result_status(item.get("compile_status")),
        cmodel_status=_result_status(item.get("cmodel_status")),
        pack_status="pass" if item.get("pack_returncode") == 0 else _result_status(item.get("pack_returncode")),
        board_status=_result_status(item.get("board_status")),
        board_all_same=_result_status(item.get("board_all_same")),
        latency_ms=_latency_from_dict(item),
        diagnosis=str(item.get("diagnosis") or ""),
    )


def _items_from_validation(path: Path, data: Dict[str, Any]) -> List[HealthItem]:
    items = []
    package_dir = str(data.get("package_dir") or path.parent)
    for item in data.get("items", []):
        if isinstance(item, dict):
            items.append(_item_from_package_validation(path, item, package_dir=package_dir))
    return items


def _item_from_failure_localization(path: Path, data: Dict[str, Any]) -> HealthItem:
    findings = data.get("findings") or []
    first = findings[0] if findings and isinstance(findings[0], dict) else {}
    return HealthItem(
        source=str(path),
        kind="failure_localization",
        name=str(data.get("model") or Path(str(data.get("result_path", path))).stem),
        board_status=str(data.get("status") or "fail"),
        diagnosis=str(first.get("likely_cause") or first.get("category") or ""),
        recommended_action=str(first.get("recommended_action") or ""),
    )


def _items_from_backlog_execution(path: Path, data: Dict[str, Any]) -> List[HealthItem]:
    items: List[HealthItem] = []
    for action in data.get("items", []):
        if not isinstance(action, dict):
            continue
        validation_json = action.get("validation_json")
        if not validation_json:
            continue
        validation_path = Path(str(validation_json))
        if not validation_path.exists():
            continue
        validation_data = _load_json(validation_path)
        if not validation_data:
            continue
        for item in _items_from_validation(validation_path, validation_data):
            items.append(
                HealthItem(
                    source=str(path),
                    kind=f"backlog_{item.kind}",
                    name=f"{action.get('target', 'unknown')}::{item.name}",
                    package_dir=item.package_dir,
                    validation_json=item.validation_json,
                    source_onnx=item.source_onnx,
                    model_name=item.model_name,
                    layout=item.layout,
                    compile_status=item.compile_status,
                    cmodel_status=item.cmodel_status,
                    pack_status=item.pack_status,
                    board_status=item.board_status,
                    board_all_same=item.board_all_same,
                    latency_ms=item.latency_ms,
                    diagnosis=item.diagnosis,
                    recommended_action=item.recommended_action,
                )
            )
    return items


def _collect_from_file(path: Path) -> List[HealthItem]:
    data = _load_json(path)
    if not data:
        return []
    name = path.name
    if name.startswith("deploy_loop_"):
        return [_item_from_deploy_loop(path, data)]
    if name.startswith("package_validation_") or name.startswith("model_validation_"):
        return _items_from_validation(path, data)
    if name.startswith("failure_localization_"):
        return [_item_from_failure_localization(path, data)]
    if name.startswith("backlog_execution_"):
        return _items_from_backlog_execution(path, data)
    return []


def _dedupe(items: Iterable[HealthItem]) -> List[HealthItem]:
    def dedupe_key(item: HealthItem) -> tuple:
        kind = item.kind.replace("backlog_", "")
        if kind == "package_item":
            name = item.name.split("::")[-1]
            return (kind, item.package_dir, name, item.source_onnx, item.model_name)
        return (item.kind, item.name, item.source)

    def item_rank(item: HealthItem) -> tuple:
        has_board = bool(item.board_status and item.board_status != "not_run")
        severity = {"fail": 3, "warn": 2, "pass": 1, "unknown": 0}.get(item.terminal_status, 0)
        return (1 if has_board else 0, severity)

    by_key: Dict[tuple, HealthItem] = {}
    for item in items:
        key = dedupe_key(item)
        previous = by_key.get(key)
        if previous is None or item_rank(item) > item_rank(previous):
            by_key[key] = item
    return list(by_key.values())


def build_health_report(roots: List[Path]) -> HealthReport:
    all_items: List[HealthItem] = []
    for root in roots:
        if root.is_file():
            all_items.extend(_collect_from_file(root))
        elif root.exists():
            for path in sorted(root.glob("*.json")):
                all_items.extend(_collect_from_file(path))
    items = _dedupe(all_items)
    by_status: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    for item in items:
        by_status[item.terminal_status] = by_status.get(item.terminal_status, 0) + 1
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
    failed = [x for x in items if x.terminal_status == "fail"]
    warn = [x for x in items if x.terminal_status == "warn"]
    pass_items = [x for x in items if x.terminal_status == "pass"]
    summary = {
        "total_items": len(items),
        "pass": len(pass_items),
        "warn": len(warn),
        "fail": len(failed),
        "unknown": by_status.get("unknown", 0),
        "by_status": by_status,
        "by_kind": by_kind,
    }
    return HealthReport(roots=[str(x) for x in roots], items=items, summary=summary)


def save_health_report_json(report: HealthReport, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "roots": report.roots,
                "summary": report.summary,
                "items": [asdict(item) | {"terminal_status": item.terminal_status} for item in report.items],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def render_health_report(report: HealthReport, limit: int = 120) -> str:
    risky = [x for x in report.items if x.terminal_status in {"fail", "warn"}]
    buckets = {
        "multi-output split required": 0,
        "board timeout / stale status": 0,
        "small-spatial high-channel conv split": 0,
        "compile or unsupported op": 0,
        "not run / incomplete validation": 0,
        "other": 0,
    }
    for item in risky:
        text = " ".join([item.board_status, item.diagnosis, item.recommended_action]).lower()
        if "exactly one output" in text or "multi-output" in text:
            buckets["multi-output split required"] += 1
        elif "small_spatial_high_channel" in text or "small-spatial" in text:
            buckets["small-spatial high-channel conv split"] += 1
        elif "timeout" in text or "stale" in text or "wr_done" in text:
            buckets["board timeout / stale status"] += 1
        elif "compile" in text or "unsupported" in text:
            buckets["compile or unsupported op"] += 1
        elif "not_run" in text or "not run" in text:
            buckets["not run / incomplete validation"] += 1
        else:
            buckets["other"] += 1
    lines = [
        "# RHB AgentFlow Health Report",
        "",
        "## Summary",
        "",
        f"- total items: `{report.summary['total_items']}`",
        f"- pass: `{report.summary['pass']}`",
        f"- warn: `{report.summary['warn']}`",
        f"- fail: `{report.summary['fail']}`",
        f"- unknown: `{report.summary['unknown']}`",
        "",
        "## Next Actions",
        "",
    ]
    for name, count in buckets.items():
        if count:
            lines.append(f"- {name}: `{count}`")
    if risky:
        lines.extend(
            [
                "",
                "Recommended focus:",
                "",
                "1. Convert repeated timeout/high-channel Conv failures into automatic IC/OC split rewrite recipes.",
                "2. Route multi-output RHB regions through the existing split-output retry executor by default.",
                "3. Promote compile-fail unsupported ops to Host or approximate-rewrite candidates before board validation.",
                "4. Re-run health-report after each validate-package/deploy-loop to check rule coverage trend.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- current accepted baseline is clean; no rewrite backlog generated.",
                "- keep archived failed probes out of the active health roots unless intentionally re-triaging them.",
                "",
            ]
        )
    lines.extend(
        [
            "## Failing Or Risky Items",
            "",
            "| status | kind | name | compile | cmodel | board | all_same | diagnosis | action |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in risky[:limit]:
        lines.append(
            "| {status} | {kind} | `{name}` | {compile} | {cmodel} | {board} | {same} | {diag} | {action} |".format(
                status=item.terminal_status,
                kind=item.kind,
                name=item.name,
                compile=item.compile_status or "-",
                cmodel=item.cmodel_status or "-",
                board=item.board_status or "-",
                same=item.board_all_same or "-",
                diag=(item.diagnosis or "-").replace("\n", " ")[:180],
                action=(item.recommended_action or "-").replace("\n", " ")[:180],
            )
        )
    if not risky:
        lines.append("| pass | - | no risky items found | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Passing Board Items",
            "",
            "| kind | name | compile | cmodel | board | all_same | latency_ms |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    passing = [x for x in report.items if x.terminal_status == "pass"]
    for item in passing[:limit]:
        latency = "-" if item.latency_ms is None else f"{item.latency_ms:.3f}"
        lines.append(
            f"| {item.kind} | `{item.name}` | {item.compile_status or '-'} | {item.cmodel_status or '-'} | "
            f"{item.board_status or '-'} | {item.board_all_same or '-'} | {latency} |"
        )
    if len(passing) > limit:
        lines.append(f"| ... | `{len(passing) - limit} more passing items omitted` | | | | | |")
    return "\n".join(lines)
