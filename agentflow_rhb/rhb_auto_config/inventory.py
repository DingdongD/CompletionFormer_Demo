from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ModelFile:
    path: Path
    name: str
    family: str
    status: str
    tags: List[str]


def infer_status(path: Path) -> str:
    parts = set(path.parts)
    if "failed_accuracy" in parts:
        return "failed_accuracy"
    if "failed_acsim" in parts:
        return "failed_compile_or_cmodel"
    if "archived_board_failed" in parts:
        return "failed_board"
    if "failed_full_channels" in parts:
        return "failed_capacity_or_channel"
    if path.name.endswith("_ckpt.py"):
        return "accepted_candidate"
    return "experiment"


def infer_family(name: str) -> str:
    prefixes = [
        "decoder",
        "head",
        "pvttiny",
        "pvt",
        "cbam",
        "backbone",
        "cspn",
        "nlspn",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            return prefix
    return "misc"


def infer_tags(name: str) -> List[str]:
    tag_keys = [
        "conv",
        "relu",
        "sigmoid",
        "fused",
        "split",
        "chunk",
        "srconv",
        "sr_down",
        "hostdown",
        "ln",
        "norm",
        "cbam",
        "mul",
        "resize",
        "exact",
        "nobias",
        "pad",
        "approx",
        "tokens",
    ]
    return [key for key in tag_keys if key in name.lower()]


def scan_models(models_root: Path) -> List[ModelFile]:
    files: List[ModelFile] = []
    for path in sorted(models_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files.append(
            ModelFile(
                path=path,
                name=path.stem,
                family=infer_family(path.name),
                status=infer_status(path),
                tags=infer_tags(path.name),
            )
        )
    return files


def summarize(files: Iterable[ModelFile]) -> Dict[str, Counter]:
    file_list = list(files)
    tag_counter: Counter = Counter()
    for item in file_list:
        tag_counter.update(item.tags)
    return {
        "status": Counter(item.status for item in file_list),
        "family": Counter(item.family for item in file_list),
        "tags": tag_counter,
    }


def render_inventory(files: List[ModelFile], limit: int = 80) -> str:
    summary = summarize(files)
    lines = [
        f"total_py_models: {len(files)}",
        "status:",
    ]
    lines.extend(f"  {key}: {value}" for key, value in sorted(summary["status"].items()))
    lines.append("family:")
    lines.extend(f"  {key}: {value}" for key, value in sorted(summary["family"].items()))
    lines.append("top_tags:")
    lines.extend(f"  {key}: {value}" for key, value in summary["tags"].most_common(20))
    lines.append("")
    lines.append(f"sample_files_first_{limit}:")
    for item in files[:limit]:
        tag_str = ",".join(item.tags) if item.tags else "-"
        lines.append(f"  {item.status:24s} {item.family:10s} {tag_str:32s} {item.path}")
    return "\n".join(lines)
