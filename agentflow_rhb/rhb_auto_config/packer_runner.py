import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PackConfig:
    compile_output_dir: str
    packer_output_dir: str
    workspace: str = "/root/demo"
    model_packer_dir: str = "Model-Packer"
    force: bool = True
    timeout_sec: int = 1200
    rram_only: bool = False


@dataclass(frozen=True)
class PackResult:
    packer_output_dir: str
    command: List[str]
    returncode: int
    has_config: bool
    has_op_insts: bool
    has_params_wei: bool
    has_params_ch: bool


def run_pack(config: PackConfig) -> PackResult:
    workspace = Path(config.workspace)
    packer_dir = workspace / config.model_packer_dir
    cmd = [
        "python",
        "main_packer.py",
        str(Path(config.packer_output_dir).resolve()),
        str(Path(config.compile_output_dir).resolve()),
    ]
    if config.force:
        cmd.append("--force")
    proc = subprocess.run(cmd, cwd=str(packer_dir), timeout=config.timeout_sec)
    out = Path(config.packer_output_dir)
    cfg = out / "config.yaml"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8")
        desired = "true" if config.rram_only else "false"
        text = text.replace("rram_only: true", f"rram_only: {desired}")
        text = text.replace("rram_only: false", f"rram_only: {desired}")
        cfg.write_text(text, encoding="utf-8")
    result = PackResult(
        packer_output_dir=str(out),
        command=cmd,
        returncode=int(proc.returncode),
        has_config=(out / "config.yaml").exists(),
        has_op_insts=(out / "op_insts_ccode.bin").exists(),
        has_params_wei=(out / "params_wei_ccode.bin").exists(),
        has_params_ch=(out / "params_ch_ccode.bin").exists(),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "rhb_pack_result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result
