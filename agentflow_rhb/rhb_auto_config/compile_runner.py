import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from log_parsers import parse_cmodel_log, parse_compile_log


@dataclass(frozen=True)
class CompileConfig:
    model: str
    workspace: str = "/root/demo"
    output_root: str = "artifacts/rhb_auto_config_framework/work/compile"
    layout: str = "input0=BWC"
    arch_path: str = "arch_16.yaml,arch_256.yaml"
    level: int = 1
    sim: int = 1
    codegen: int = 3
    split: int = 32
    concat: int = 100
    seed: int = 1
    run_cv_model: bool = True
    timeout_sec: int = 1800
    checkpoint: str = ""


@dataclass(frozen=True)
class CompileResult:
    model: str
    output_dir: str
    commands: List[str]
    cv_returncode: Optional[int]
    compile_returncode: Optional[int]
    cmodel_returncode: Optional[int]
    compile: dict
    cmodel: dict


def _run(cmd: List[str], cwd: Path, timeout: int, env: dict) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), timeout=timeout, env=env)
    return int(proc.returncode)


def run_compile_cmodel(config: CompileConfig) -> CompileResult:
    workspace = Path(config.workspace)
    output_dir = workspace / config.output_root / config.model
    (workspace / config.output_root).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands: List[str] = []
    env = os.environ.copy()
    if config.checkpoint:
        env["COMPLETIONFORMER_HW_CKPT"] = config.checkpoint
    cv_rc: Optional[int] = None
    if config.run_cv_model:
        cmd = [
            "make",
            f"model={config.model}",
            f"output_root={config.output_root}",
            "cv_model",
        ]
        commands.append(" ".join(cmd))
        cv_rc = _run(cmd, workspace, config.timeout_sec, env)
    compile_cmd = [
        "make",
        f"model={config.model}",
        f"output_root={config.output_root}",
        f"layout={config.layout}",
        f"arch_path={config.arch_path}",
        f"level={config.level}",
        f"sim={config.sim}",
        f"codegen={config.codegen}",
        f"split={config.split}",
        f"concat={config.concat}",
        "compile",
    ]
    commands.append(" ".join(compile_cmd))
    compile_rc = _run(compile_cmd, workspace, config.timeout_sec, env)
    cmodel_cmd = [
        "make",
        f"model={config.model}",
        f"output_root={config.output_root}",
        f"arch_path={config.arch_path}",
        f"seed={config.seed}",
        "cmodel",
    ]
    commands.append(" ".join(cmodel_cmd))
    cmodel_rc = _run(cmodel_cmd, workspace, config.timeout_sec, env)
    result = CompileResult(
        model=config.model,
        output_dir=str(output_dir),
        commands=commands,
        cv_returncode=cv_rc,
        compile_returncode=compile_rc,
        cmodel_returncode=cmodel_rc,
        compile=parse_compile_log(output_dir / "compile.log"),
        cmodel=parse_cmodel_log(output_dir / "cmodel.log"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rhb_compile_result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def render_compile_result(result: CompileResult) -> str:
    return "\n".join(
        [
            f"model: {result.model}",
            f"output_dir: {result.output_dir}",
            f"cv_returncode: {result.cv_returncode}",
            f"compile_returncode: {result.compile_returncode}",
            f"compile_status: {result.compile.get('status')}",
            f"cmodel_returncode: {result.cmodel_returncode}",
            f"cmodel_status: {result.cmodel.get('status')}",
        ]
    )
