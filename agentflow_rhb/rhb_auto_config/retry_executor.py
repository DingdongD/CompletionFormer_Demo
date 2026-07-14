import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from submodel_exporter import ExportedSubmodel, export_rhb_submodels


@dataclass(frozen=True)
class RetryExecution:
    retry_plan: str
    executed_actions: List[str]
    exported_submodels: List[ExportedSubmodel]
    notes: List[str]


def execute_retry_plan(retry_plan_json: Path, region_plan_json: Path, output_dir: Path) -> RetryExecution:
    plan = json.loads(retry_plan_json.read_text(encoding="utf-8"))
    actions = [item.get("action", "") for item in plan.get("actions", [])]
    executed: List[str] = []
    exported: List[ExportedSubmodel] = []
    notes: List[str] = []

    if "split_multi_output_region" in actions:
        exported = export_rhb_submodels(region_plan_json, output_dir, split_multi_output=True)
        executed.append("split_multi_output_region")
        notes.append("exported original RHB regions plus one-output variants for multi-output regions")

    unsupported = [action for action in actions if action not in {"split_multi_output_region"}]
    for action in unsupported:
        notes.append(f"action '{action}' is planned but not automatically executable yet")

    if not executed:
        notes.append("no executable retry action found")

    return RetryExecution(
        retry_plan=str(retry_plan_json),
        executed_actions=executed,
        exported_submodels=exported,
        notes=notes,
    )


def save_retry_execution(execution: RetryExecution, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(execution), indent=2), encoding="utf-8")


def render_retry_execution(execution: RetryExecution) -> str:
    lines = [
        "# Retry Execution",
        "",
        f"- retry plan: `{execution.retry_plan}`",
        f"- executed actions: {execution.executed_actions}",
        "",
        "## Exported Submodels",
        "",
        "region_id\tstatus\tonnx_path\tinputs\toutputs",
    ]
    for item in execution.exported_submodels:
        lines.append(f"{item.region_id}\t{item.status}\t{item.onnx_path}\t{item.input_names}\t{item.output_names}")
    lines.extend(["", "## Notes", ""])
    for note in execution.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)
