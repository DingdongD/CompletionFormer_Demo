import re
from pathlib import Path
from typing import Dict, List


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_compile_log(path: Path) -> Dict[str, object]:
    text = read_text(path)
    lower = text.lower()
    out_dir = path.parent
    cfg_files = list(out_dir.glob("*_cfg.txt"))
    op_bins = list(out_dir.glob("*_op_insts_ccode.bin"))
    graph_dots = list(out_dir.glob("*_ACPC_IR.dot"))
    has_key_artifacts = bool(cfg_files and op_bins and graph_dots)
    errors = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"(fatal|failed|exception|traceback|unsupported|segmentation fault|assertion|aborted)", line, re.IGNORECASE)
    ]
    status = "pass"
    if not text:
        status = "missing_log"
    elif errors:
        status = "fail"
    elif has_key_artifacts:
        status = "pass"
    elif "error" in lower:
        status = "pass_with_warnings"
    return {
        "status": status,
        "has_key_artifacts": has_key_artifacts,
        "error_count": len(errors),
        "errors": errors[:40],
        "log": str(path),
    }


def parse_cmodel_log(path: Path) -> Dict[str, object]:
    text = read_text(path)
    lower = text.lower()
    markers = [
        "completed the simulation",
        "simulation completed",
        "threadid 0 completed",
    ]
    pass_marker = any(marker in lower for marker in markers)
    errors = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"(error|failed|exception|traceback|mismatch)", line, re.IGNORECASE)
    ]
    if not text:
        status = "missing_log"
    elif pass_marker and not errors:
        status = "pass"
    elif pass_marker:
        status = "pass_with_warnings"
    else:
        status = "fail"
    return {
        "status": status,
        "pass_marker": pass_marker,
        "error_count": len(errors),
        "errors": errors[:40],
        "log": str(path),
    }


def parse_board_log(path: Path) -> Dict[str, object]:
    text = read_text(path)
    all_same = None
    m = re.search(r"All same:\s*(True|False)", text)
    if m:
        all_same = m.group(1) == "True"
    timeouts = re.findall(r"(timeout|wait too long|Problem for wait interrupt)", text, re.IGNORECASE)
    errors = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"(error|failed|timeout|wait too long|Problem for wait interrupt)", line, re.IGNORECASE)
    ]
    latency = {}
    for label, value in re.findall(r"LATENCY\s+([^:]+):\s*([0-9.]+)\s*ms", text):
        latency[label.strip()] = float(value)
    counters = {}
    for key, value in re.findall(r"(sta_[a-zA-Z0-9_]+)\s*[:=]\s*([0-9]+)", text):
        counters[key] = int(value)
    if not text:
        status = "missing_log"
    elif timeouts:
        status = "fail_timeout"
    elif all_same is False:
        status = "fail_accuracy"
    elif all_same is True or latency:
        status = "pass"
    elif errors:
        status = "fail_runtime"
    else:
        status = "unknown"
    return {
        "status": status,
        "all_same": all_same,
        "error_count": len(errors),
        "errors": errors[:60],
        "latency": latency,
        "counters": counters,
        "log": str(path),
    }
