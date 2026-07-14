import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from board_runner import BoardConfig, run_board
from compile_runner import CompileConfig, run_compile_cmodel
from packer_runner import PackConfig, run_pack


@dataclass(frozen=True)
class DeployLoopConfig:
    model: str
    workspace: str = "/root/demo"
    work_root: str = "artifacts/rhb_auto_config_framework/work"
    layout: str = "input0=BWC"
    arch_path: str = "arch_16.yaml,arch_256.yaml"
    run_cv_model: bool = True
    run_board: bool = False
    board: str = "root@192.168.115.122"
    board_password: str = "root"
    board_work_dir: str = "/home/root/workspace/demo_vp_xj/packers/rhb_auto_probe"
    local_runner: str = ""
    runner_args: str = ""
    checkpoint: str = ""


def run_deploy_loop(config: DeployLoopConfig) -> dict:
    workspace = Path(config.workspace)
    compile_root = f"{config.work_root}/compile"
    pack_root = workspace / config.work_root / "packer"
    compile_result = run_compile_cmodel(
        CompileConfig(
            model=config.model,
            workspace=config.workspace,
            output_root=compile_root,
            layout=config.layout,
            arch_path=config.arch_path,
            run_cv_model=config.run_cv_model,
            checkpoint=config.checkpoint,
        )
    )
    compile_status = compile_result.compile.get("status")
    cmodel_status = compile_result.cmodel.get("status")
    if compile_status != "pass" or cmodel_status not in {"pass", "pass_with_warnings"}:
        result = {
            "model": config.model,
            "compile": asdict(compile_result),
            "pack": None,
            "board": None,
            "skipped": "pack_and_board skipped because compile/cmodel did not pass",
        }
        out_dir = workspace / config.work_root / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = config.model.replace("/", ".")
        (out_dir / f"deploy_loop_{safe_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    packer_dir = pack_root / f"packer_{config.model}"
    pack_result = run_pack(
        PackConfig(
            workspace=config.workspace,
            compile_output_dir=str(workspace / compile_root / config.model),
            packer_output_dir=str(packer_dir),
        )
    )
    board_result = None
    if config.run_board:
        board_result = run_board(
            BoardConfig(
                packer_dir=str(packer_dir),
                board=config.board,
                password=config.board_password,
                board_work_dir=config.board_work_dir,
                local_runner=config.local_runner,
                runner_args=config.runner_args.split() if config.runner_args else [],
                log_path=str(packer_dir / "board_test.log"),
            )
        )
    result = {
        "model": config.model,
        "compile": asdict(compile_result),
        "pack": asdict(pack_result),
        "board": asdict(board_result) if board_result else None,
    }
    out_dir = workspace / config.work_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = config.model.replace("/", ".")
    (out_dir / f"deploy_loop_{safe_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def render_deploy_loop(result: dict) -> str:
    lines = [
        f"model: {result['model']}",
        f"compile_status: {result['compile']['compile'].get('status')}",
        f"cmodel_status: {result['compile']['cmodel'].get('status')}",
    ]
    if result.get("pack") is None:
        lines.append(f"skipped: {result.get('skipped')}")
        return "\n".join(lines)
    lines.extend(
        [
            f"pack_returncode: {result['pack']['returncode']}",
            f"pack_has_config: {result['pack']['has_config']}",
        ]
    )
    if result.get("board"):
        lines.extend(
            [
                f"board_returncode: {result['board']['returncode']}",
                f"board_status: {result['board']['parsed'].get('status')}",
                f"board_all_same: {result['board']['parsed'].get('all_same')}",
            ]
        )
    return "\n".join(lines)
