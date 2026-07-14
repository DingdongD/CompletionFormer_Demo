import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from pipeline import DeployLoopConfig, run_deploy_loop


@dataclass(frozen=True)
class ModelValidationItem:
    model: str
    compile_status: str
    cmodel_status: str
    pack_returncode: int
    board_status: str
    board_all_same: str
    result_json: str


@dataclass(frozen=True)
class ModelValidationReport:
    models: List[str]
    run_board: bool
    items: List[ModelValidationItem]


def validate_models(
    models: List[str],
    layout: str,
    workspace: str,
    work_root: str,
    checkpoint: str = "",
    run_board: bool = False,
    skip_cv_model: bool = True,
    board: str = "root@192.168.115.122",
    password: str = "root",
    board_work_dir: str = "/home/root/workspace/demo_vp_xj/packers/rhb_auto_probe",
) -> ModelValidationReport:
    items: List[ModelValidationItem] = []
    for model in models:
        result = run_deploy_loop(
            DeployLoopConfig(
                model=model,
                workspace=workspace,
                work_root=work_root,
                layout=layout,
                run_cv_model=not skip_cv_model,
                run_board=run_board,
                board=board,
                board_password=password,
                board_work_dir=board_work_dir,
                checkpoint=checkpoint,
            )
        )
        compile_status = result["compile"]["compile"].get("status", "unknown") if result.get("compile") else "not_run"
        cmodel_status = result["compile"]["cmodel"].get("status", "unknown") if result.get("compile") else "not_run"
        board_status = "not_run"
        board_all_same = ""
        if result.get("board"):
            board_status = str(result["board"]["parsed"].get("status", "unknown"))
            board_all_same = str(result["board"]["parsed"].get("all_same", ""))
        items.append(
            ModelValidationItem(
                model=model,
                compile_status=compile_status,
                cmodel_status=cmodel_status,
                pack_returncode=result["pack"]["returncode"] if result.get("pack") else -999,
                board_status=board_status,
                board_all_same=board_all_same,
                result_json=str(Path(work_root) / "reports" / f"deploy_loop_{model}.json"),
            )
        )
    return ModelValidationReport(models=models, run_board=run_board, items=items)


def save_validation_report(report: ModelValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def render_validation_report(report: ModelValidationReport) -> str:
    lines = [
        "# Model Validation Report",
        "",
        f"- models: {len(report.models)}",
        f"- run_board: {report.run_board}",
        "",
        "model\tcompile\tcmodel\tpack_rc\tboard\tall_same\tresult_json",
    ]
    for item in report.items:
        lines.append(
            f"{item.model}\t{item.compile_status}\t{item.cmodel_status}\t{item.pack_returncode}\t"
            f"{item.board_status}\t{item.board_all_same}\t{item.result_json}"
        )
    return "\n".join(lines)
