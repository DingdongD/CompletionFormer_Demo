import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class RewriteAction:
    priority: int
    category: str
    target: str
    exact: bool
    confidence: str
    reason: str
    suggested_action: str
    command_hint: str = ""
    source: str = ""


@dataclass(frozen=True)
class RewriteBacklog:
    health_json: str
    actions: List[RewriteAction]
    summary: Dict[str, int]


def _load_health_items(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data.get("items", []) if isinstance(item, dict)]


def _action_for_item(item: Dict[str, Any]) -> RewriteAction:
    name = str(item.get("name") or "unknown")
    source = str(item.get("source") or "")
    diagnosis = str(item.get("diagnosis") or "")
    action = str(item.get("recommended_action") or "")
    board_status = str(item.get("board_status") or "")
    text = " ".join([name, diagnosis, action, board_status]).lower()
    source_onnx = str(item.get("source_onnx") or "")
    target_safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)

    if "exactly one output" in text or "multi-output" in text:
        return RewriteAction(
            priority=1,
            category="split_multi_output",
            target=name,
            exact=True,
            confidence="high",
            reason="board runtime accepts one output tensor per submodel",
            suggested_action="Split this RHB region into one exported ONNX per output and keep fan-out glue on Host.",
            command_hint=(
                "python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py plan-retry "
                f"--localization-json {source}"
            )
            if "failure_localization_" in Path(source).name
            else "",
            source=source,
        )

    if "small_spatial_high_channel" in text or "small-spatial high-channel" in text:
        return RewriteAction(
            priority=2,
            category="conv_ic_oc_split",
            target=name,
            exact=True,
            confidence="high",
            reason="small spatial high-channel Conv repeatedly times out on board",
            suggested_action=(
                "Apply exact input-channel/output-channel Conv splitting. Prefer IC chunk <= 8 first; "
                "Host or board CPU accumulates partial sums for each output-channel chunk. "
                "Then run validate-package on the generated split package."
            ),
            command_hint=(
                "python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py export-conv-splits "
                f"--onnx {source_onnx} --node-index 0 "
                "--mode input-output --output-chunk-channels 32 --input-chunk-channels 8 "
                f"--output-dir artifacts/rhb_auto_config_framework/work/rewrite_splits/{target_safe}"
            )
            if source_onnx
            else "",
            source=source,
        )

    if "timeout" in text or "stale" in text or "wr_done" in text:
        return RewriteAction(
            priority=3,
            category="timeout_bisect",
            target=name,
            exact=True,
            confidence="medium",
            reason="board status wait timeout or stale runtime state",
            suggested_action=(
                "Retry with rram_only=false and clear-wr-done. If still failing, binary-split the region or apply "
                "Conv IC/OC splitting when Conv dominates."
            ),
            command_hint=(
                "python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py localize-failure "
                f"--result-json {source}"
            )
            if "deploy_loop_" in Path(source).name
            else "",
            source=source,
        )

    if "compile" in text or "unsupported" in text:
        return RewriteAction(
            priority=4,
            category="unsupported_op_fallback_or_rewrite",
            target=name,
            exact=False,
            confidence="medium",
            reason="compiler rejected the graph or an unsupported op pattern was detected",
            suggested_action=(
                "Keep the unsupported op on Host if exactness is required. If retraining is allowed, apply the "
                "hardware-aligned replacement rule and regenerate calibration."
            ),
            command_hint="python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py suggest-rule --result-json <result.json>",
            source=source,
        )

    if "not_run" in text or "not run" in text:
        package_dir = str(item.get("package_dir") or "")
        command_hint = (
            "python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py validate-package "
            f"--package-dir {package_dir} --run-board"
            if package_dir
            else "python artifacts/rhb_auto_config_framework/rhb_auto_config/cli.py validate-package --package-dir <pkg> --run-board"
        )
        return RewriteAction(
            priority=5,
            category="complete_validation",
            target=name,
            exact=True,
            confidence="medium",
            reason="submodel has not completed board validation",
            suggested_action="Run validate-package/deploy-loop with --run-board before promoting this region.",
            command_hint=command_hint,
            source=source,
        )

    return RewriteAction(
        priority=9,
        category="manual_triage",
        target=name,
        exact=True,
        confidence="low",
        reason=diagnosis or "no known automatic rewrite matched",
        suggested_action="Inspect compile/cmodel/board logs and add a new rule if the failure repeats.",
        source=source,
    )


def build_rewrite_backlog(health_json: Path) -> RewriteBacklog:
    items = _load_health_items(health_json)
    risky = [item for item in items if item.get("terminal_status") in {"fail", "warn"}]
    actions = [_action_for_item(item) for item in risky]
    # Keep the first action for duplicate target/category/source triples.
    deduped: List[RewriteAction] = []
    seen = set()
    for action in sorted(actions, key=lambda x: (x.priority, x.category, x.target)):
        key = (action.category, action.target, action.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    summary: Dict[str, int] = {}
    for action in deduped:
        summary[action.category] = summary.get(action.category, 0) + 1
    return RewriteBacklog(health_json=str(health_json), actions=deduped, summary=summary)


def save_rewrite_backlog_json(backlog: RewriteBacklog, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "health_json": backlog.health_json,
                "summary": backlog.summary,
                "actions": [asdict(action) for action in backlog.actions],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def render_rewrite_backlog(backlog: RewriteBacklog, limit: int = 120) -> str:
    lines = [
        "# RHB AgentFlow Rewrite Backlog",
        "",
        f"- health_json: `{backlog.health_json}`",
        "",
        "## Summary",
        "",
    ]
    for key, count in sorted(backlog.summary.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {key}: `{count}`")
    lines.extend(
        [
            "",
            "## Actions",
            "",
            "| priority | category | target | exact | confidence | reason | suggested action | command hint |",
            "|---:|---|---|---|---|---|---|---|",
        ]
    )
    for action in backlog.actions[:limit]:
        lines.append(
            "| {priority} | {category} | `{target}` | {exact} | {confidence} | {reason} | {suggested} | `{command}` |".format(
                priority=action.priority,
                category=action.category,
                target=action.target,
                exact=str(action.exact).lower(),
                confidence=action.confidence,
                reason=action.reason.replace("\n", " ")[:160],
                suggested=action.suggested_action.replace("\n", " ")[:220],
                command=(action.command_hint or "-").replace("|", "\\|")[:240],
            )
        )
    if len(backlog.actions) > limit:
        lines.append(f"| ... | ... | `{len(backlog.actions) - limit} more actions omitted` | | | | | |")
    return "\n".join(lines)
