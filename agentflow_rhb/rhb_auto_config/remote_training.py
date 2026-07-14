import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RemoteStageFile:
    local: str
    remote: str


@dataclass(frozen=True)
class RemoteTrainingProfile:
    name: str
    description: str
    user: str
    host: str
    port: int
    identity_file: str
    ssh_options: List[str]
    remote_cwd: str
    remote_stage_dir: str
    remote_output_dir: str
    local_fetch_dir: str
    stage_files: List[RemoteStageFile]
    train_command: str
    ckpt_patterns: List[str]


@dataclass(frozen=True)
class RemoteTrainingPlan:
    profile: RemoteTrainingProfile
    generated_script: str
    ssh_base: str
    stage_commands: List[str]
    submit_command: str
    status_command: str
    fetch_commands: List[str]


@dataclass(frozen=True)
class RemoteTrainingResult:
    action: str
    profile: str
    ok: bool
    message: str
    generated_script: str
    log: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_user(path: str) -> str:
    return str(Path(path).expanduser())


def _ssh_args(profile: RemoteTrainingProfile) -> List[str]:
    args = [
        "ssh",
        "-i",
        _expand_user(profile.identity_file),
        "-p",
        str(profile.port),
    ]
    args.extend(profile.ssh_options)
    args.append(f"{profile.user}@{profile.host}")
    return args


def _scp_args(profile: RemoteTrainingProfile) -> List[str]:
    args = [
        "scp",
        "-i",
        _expand_user(profile.identity_file),
        "-P",
        str(profile.port),
        "-r",
    ]
    args.extend(profile.ssh_options)
    return args


def _ssh_base_string(profile: RemoteTrainingProfile) -> str:
    return " ".join(shlex.quote(x) for x in _ssh_args(profile))


def _remote_spec(profile: RemoteTrainingProfile, path: str) -> str:
    return f"{profile.user}@{profile.host}:{path}"


def _remote_target(profile: RemoteTrainingProfile, target: str) -> str:
    if target.startswith("/"):
        return target
    return f"{profile.remote_stage_dir.rstrip('/')}/{target}"


def _format_command(profile: RemoteTrainingProfile, command: str) -> str:
    return command.format(
        remote_cwd=profile.remote_cwd,
        remote_stage_dir=profile.remote_stage_dir,
        remote_output_dir=profile.remote_output_dir,
        profile=profile.name,
    )


def load_remote_training_profile(config_path: Path, profile_name: str) -> RemoteTrainingProfile:
    data = _load_json(config_path)
    defaults = data.get("default_ssh", {})
    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        known = ", ".join(sorted(profiles))
        raise KeyError(f"unknown remote training profile: {profile_name}; known: {known}")
    profile = profiles[profile_name]
    merged = {**defaults, **profile}
    return RemoteTrainingProfile(
        name=profile_name,
        description=str(merged.get("description", "")),
        user=str(merged["user"]),
        host=str(merged["host"]),
        port=int(merged["port"]),
        identity_file=str(merged["identity_file"]),
        ssh_options=[str(x) for x in merged.get("ssh_options", [])],
        remote_cwd=str(merged["remote_cwd"]),
        remote_stage_dir=str(merged["remote_stage_dir"]),
        remote_output_dir=str(merged["remote_output_dir"]),
        local_fetch_dir=str(merged["local_fetch_dir"]),
        stage_files=[RemoteStageFile(local=str(x["local"]), remote=str(x["remote"])) for x in merged.get("stage_files", [])],
        train_command=str(merged["train_command"]),
        ckpt_patterns=[str(x) for x in merged.get("ckpt_patterns", [])],
    )


def render_remote_script(profile: RemoteTrainingProfile) -> str:
    train_command = _format_command(profile, profile.train_command)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(profile.remote_cwd)}",
        f"mkdir -p {shlex.quote(profile.remote_output_dir)}",
        f"echo '[agentflow] profile={profile.name}'",
        "echo '[agentflow] start='$(date -Is)",
        "nvidia-smi || true",
        "python - <<'PY'",
        "import torch",
        "print('torch', torch.__version__)",
        "print('cuda_available', torch.cuda.is_available())",
        "print('cuda_device_count', torch.cuda.device_count())",
        "PY",
        train_command,
        "echo '[agentflow] done='$(date -Is)",
    ]
    return "\n".join(lines) + "\n"


def build_remote_training_plan(
    config_path: Path,
    profile_name: str,
    report_dir: Path,
) -> RemoteTrainingPlan:
    profile = load_remote_training_profile(config_path, profile_name)
    report_dir.mkdir(parents=True, exist_ok=True)
    script_path = report_dir / f"remote_train_{profile.name}.sh"
    script_path.write_text(render_remote_script(profile), encoding="utf-8")
    stage_commands: List[str] = []
    mkdir_cmd = f"mkdir -p {shlex.quote(profile.remote_stage_dir)} {shlex.quote(profile.remote_output_dir)}"
    stage_commands.append(f"{_ssh_base_string(profile)} {shlex.quote(mkdir_cmd)}")
    for item in profile.stage_files:
        remote = _remote_target(profile, item.remote)
        remote_parent = str(Path(remote).parent)
        stage_commands.append(f"{_ssh_base_string(profile)} {shlex.quote('mkdir -p ' + shlex.quote(remote_parent))}")
        scp = _scp_args(profile) + [item.local, _remote_spec(profile, remote)]
        stage_commands.append(" ".join(shlex.quote(x) for x in scp))
    remote_script = f"{profile.remote_stage_dir.rstrip('/')}/run_train.sh"
    scp_script = _scp_args(profile) + [str(script_path), _remote_spec(profile, remote_script)]
    stage_commands.append(" ".join(shlex.quote(x) for x in scp_script))
    submit_inner = (
        f"cd {shlex.quote(profile.remote_cwd)} && "
        f"mkdir -p {shlex.quote(profile.remote_output_dir)} && "
        f"nohup bash {shlex.quote(remote_script)} > {shlex.quote(profile.remote_output_dir + '/agentflow_train.log')} "
        f"2>&1 < /dev/null & echo $! > {shlex.quote(profile.remote_output_dir + '/agentflow_train.pid')} && "
        f"cat {shlex.quote(profile.remote_output_dir + '/agentflow_train.pid')}"
    )
    status_inner = (
        f"set +e; pid_file={shlex.quote(profile.remote_output_dir + '/agentflow_train.pid')}; "
        "if [ -f \"$pid_file\" ]; then pid=$(cat \"$pid_file\"); ps -p \"$pid\" -o pid,etime,cmd; else echo 'no pid file'; fi; "
        f"echo '--- train log tail ---'; tail -n 80 {shlex.quote(profile.remote_output_dir + '/log_train.txt')} 2>/dev/null || "
        f"tail -n 80 {shlex.quote(profile.remote_output_dir + '/agentflow_train.log')} 2>/dev/null || true"
    )
    fetch_commands = []
    for pattern in profile.ckpt_patterns:
        local_dir = Path(profile.local_fetch_dir)
        remote_pattern = f"{profile.remote_output_dir.rstrip('/')}/{pattern}"
        scp = _scp_args(profile) + [_remote_spec(profile, remote_pattern), str(local_dir) + "/"]
        fetch_commands.append(" ".join(shlex.quote(x) for x in scp))
    return RemoteTrainingPlan(
        profile=profile,
        generated_script=str(script_path),
        ssh_base=_ssh_base_string(profile),
        stage_commands=stage_commands,
        submit_command=f"{_ssh_base_string(profile)} {shlex.quote(submit_inner)}",
        status_command=f"{_ssh_base_string(profile)} {shlex.quote(status_inner)}",
        fetch_commands=fetch_commands,
    )


def save_remote_training_plan(plan: RemoteTrainingPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")


def render_remote_training_plan(plan: RemoteTrainingPlan) -> str:
    lines = [
        "# Remote Training Plan",
        "",
        f"- profile: `{plan.profile.name}`",
        f"- description: {plan.profile.description}",
        f"- remote: `{plan.profile.user}@{plan.profile.host}:{plan.profile.port}`",
        f"- remote_cwd: `{plan.profile.remote_cwd}`",
        f"- remote_stage_dir: `{plan.profile.remote_stage_dir}`",
        f"- remote_output_dir: `{plan.profile.remote_output_dir}`",
        f"- local_fetch_dir: `{plan.profile.local_fetch_dir}`",
        f"- generated_script: `{plan.generated_script}`",
        "",
        "## Stage Commands",
        "",
    ]
    lines.extend(f"```bash\n{cmd}\n```" for cmd in plan.stage_commands)
    lines.extend(["", "## Submit Command", "", f"```bash\n{plan.submit_command}\n```"])
    lines.extend(["", "## Status Command", "", f"```bash\n{plan.status_command}\n```"])
    lines.extend(["", "## Fetch Commands", ""])
    lines.extend(f"```bash\n{cmd}\n```" for cmd in plan.fetch_commands)
    return "\n".join(lines)


def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def _run_ssh(profile: RemoteTrainingProfile, remote_command: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return _run(_ssh_args(profile) + [remote_command], timeout=timeout)


def submit_remote_training(plan: RemoteTrainingPlan, timeout: int = 120) -> RemoteTrainingResult:
    profile = plan.profile
    logs = []
    rc_ok = True
    for cmd_text in plan.stage_commands:
        proc = _run(["bash", "-lc", cmd_text], timeout=timeout)
        logs.append(f"$ {cmd_text}\n{proc.stdout}")
        rc_ok = rc_ok and proc.returncode == 0
        if proc.returncode != 0:
            return RemoteTrainingResult("submit", profile.name, False, "stage failed", plan.generated_script, "\n".join(logs))
    remote_script = f"{profile.remote_stage_dir.rstrip('/')}/run_train.sh"
    submit_inner = (
        f"cd {shlex.quote(profile.remote_cwd)} && "
        f"mkdir -p {shlex.quote(profile.remote_output_dir)} && "
        f"nohup bash {shlex.quote(remote_script)} > {shlex.quote(profile.remote_output_dir + '/agentflow_train.log')} "
        f"2>&1 < /dev/null & echo $! > {shlex.quote(profile.remote_output_dir + '/agentflow_train.pid')} && "
        f"cat {shlex.quote(profile.remote_output_dir + '/agentflow_train.pid')}"
    )
    proc = _run_ssh(profile, submit_inner, timeout=timeout)
    logs.append(f"$ submit\n{proc.stdout}")
    ok = rc_ok and proc.returncode == 0
    return RemoteTrainingResult("submit", profile.name, ok, "submitted" if ok else "submit failed", plan.generated_script, "\n".join(logs))


def status_remote_training(profile: RemoteTrainingProfile, timeout: int = 60) -> RemoteTrainingResult:
    status_inner = (
        f"set +e; pid_file={shlex.quote(profile.remote_output_dir + '/agentflow_train.pid')}; "
        "if [ -f \"$pid_file\" ]; then pid=$(cat \"$pid_file\"); ps -p \"$pid\" -o pid,etime,cmd; else echo 'no pid file'; fi; "
        f"echo '--- train log tail ---'; tail -n 80 {shlex.quote(profile.remote_output_dir + '/log_train.txt')} 2>/dev/null || "
        f"tail -n 80 {shlex.quote(profile.remote_output_dir + '/agentflow_train.log')} 2>/dev/null || true"
    )
    proc = _run_ssh(profile, status_inner, timeout=timeout)
    return RemoteTrainingResult("status", profile.name, proc.returncode == 0, "status read" if proc.returncode == 0 else "status failed", "", proc.stdout)


def fetch_remote_training(profile: RemoteTrainingProfile, timeout: int = 300) -> RemoteTrainingResult:
    local_dir = Path(profile.local_fetch_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    ok_any = False
    for pattern in profile.ckpt_patterns:
        remote_pattern = f"{profile.remote_output_dir.rstrip('/')}/{pattern}"
        cmd = _scp_args(profile) + [_remote_spec(profile, remote_pattern), str(local_dir) + "/"]
        proc = _run(cmd, timeout=timeout)
        logs.append(f"$ {' '.join(shlex.quote(x) for x in cmd)}\n{proc.stdout}")
        ok_any = ok_any or proc.returncode == 0
    message = f"fetched to {local_dir}" if ok_any else "no artifacts fetched"
    return RemoteTrainingResult("fetch", profile.name, ok_any, message, "", "\n".join(logs))


def stop_remote_training(profile: RemoteTrainingProfile, timeout: int = 60) -> RemoteTrainingResult:
    stop_inner = (
        f"pid_file={shlex.quote(profile.remote_output_dir + '/agentflow_train.pid')}; "
        "if [ -f \"$pid_file\" ]; then pid=$(cat \"$pid_file\"); kill \"$pid\" 2>/dev/null || true; echo stopped $pid; "
        "else echo 'no pid file'; fi"
    )
    proc = _run_ssh(profile, stop_inner, timeout=timeout)
    return RemoteTrainingResult("stop", profile.name, proc.returncode == 0, "stop requested" if proc.returncode == 0 else "stop failed", "", proc.stdout)


def timestamped_name(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
