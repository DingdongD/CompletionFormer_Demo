import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from log_parsers import parse_board_log


@dataclass(frozen=True)
class BoardConfig:
    packer_dir: str
    board: str = "root@192.168.115.122"
    password: str = "root"
    board_work_dir: str = "/home/root/workspace/demo_vp_xj/packers/rhb_auto_probe"
    runner: str = "/home/root/workspace/demo_vp_xj/deploy.py"
    model_name: str = ""
    local_runner: str = ""
    runner_args: List[str] = None
    log_path: str = ""
    timeout_sec: int = 1800


@dataclass(frozen=True)
class BoardResult:
    packer_dir: str
    board: str
    commands: List[str]
    returncode: int
    parsed: dict


def _run_shell(cmd: str, timeout: int, log_file: Optional[Path] = None) -> int:
    if log_file is None:
        proc = subprocess.run(cmd, shell=True, timeout=timeout)
        return int(proc.returncode)
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, shell=True, timeout=timeout, stdout=f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def run_board(config: BoardConfig) -> BoardResult:
    runner_args = config.runner_args or []
    packer = Path(config.packer_dir)
    name = packer.name
    remote_packer = f"{config.board_work_dir}/{name}"
    log_path = Path(config.log_path) if config.log_path else packer / "board_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    commands = [
        f"sshpass -p '{config.password}' ssh {ssh_opts} {config.board} \"rm -rf '{config.board_work_dir}' && mkdir -p '{config.board_work_dir}'\"",
        f"sshpass -p '{config.password}' scp {ssh_opts} -r '{packer}' {config.board}:'{config.board_work_dir}/'",
    ]
    for cmd in commands:
        rc = _run_shell(cmd, config.timeout_sec)
        if rc != 0:
            parsed = {"status": "fail_transfer", "error_count": 1, "errors": [f"command failed: {cmd}"]}
            return BoardResult(str(packer), config.board, commands, rc, parsed)
    if config.local_runner:
        local_runner = Path(config.local_runner)
        commands.append(
            f"sshpass -p '{config.password}' scp {ssh_opts} '{local_runner}' {config.board}:'{config.board_work_dir}/'"
        )
        rc = _run_shell(commands[-1], config.timeout_sec)
        if rc != 0:
            parsed = {"status": "fail_runner_transfer", "error_count": 1, "errors": [f"command failed: {commands[-1]}"]}
            return BoardResult(str(packer), config.board, commands, rc, parsed)
        remote_runner = f"{config.board_work_dir}/{local_runner.name}"
        run_cmd = f"cd '{config.board_work_dir}' && python3 '{remote_runner}' '{remote_packer}' {' '.join(runner_args)}"
    else:
        model_name = config.model_name or name.replace("packer_", "")
        run_cmd = f"cd '{config.board_work_dir}' && python3 '{config.runner}' '{name}/' '{model_name}' {' '.join(runner_args)}"
    board_cmd = f"sshpass -p '{config.password}' ssh {ssh_opts} {config.board} \"{run_cmd}\""
    commands.append(board_cmd)
    rc = _run_shell(board_cmd, config.timeout_sec, log_path)
    parsed = parse_board_log(log_path)
    result = BoardResult(str(packer), config.board, commands, rc, parsed)
    (log_path.parent / "rhb_board_result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def render_board_result(result: BoardResult) -> str:
    lines = [
        f"packer_dir: {result.packer_dir}",
        f"board: {result.board}",
        f"returncode: {result.returncode}",
        f"board_status: {result.parsed.get('status')}",
        f"all_same: {result.parsed.get('all_same')}",
        f"error_count: {result.parsed.get('error_count')}",
    ]
    latency = result.parsed.get("latency") or {}
    if latency:
        lines.append("latency:")
        lines.extend(f"  {key}: {value:.3f} ms" for key, value in latency.items())
    return "\n".join(lines)
