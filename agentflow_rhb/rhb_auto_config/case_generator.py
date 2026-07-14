import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class GeneratedCase:
    case_name: str
    model_family: str
    input_shape: List[int]
    source_root: str
    onnx_glob: str
    checkpoint: str
    representative_data: str
    deployment_goal: str
    notes: List[str]


def generate_case(
    case_name: str,
    model_family: str,
    source_root: Path,
    input_shape: List[int],
    onnx_glob: str = "",
    checkpoint: str = "",
    representative_data: str = "",
    deployment_goal: str = "maximize board-pass RHB subgraphs while preserving exact Host fallback for risky glue",
) -> GeneratedCase:
    notes = [
        "Run profile-source first to identify high-risk operators in the reference code.",
        "Run score-onnx-dir if probe/exported ONNX files already exist.",
        "Use optimize-onnx --export-submodels for each candidate full/subgraph ONNX.",
        "Only promote rules after board pass; CModel pass alone is not sufficient.",
    ]
    return GeneratedCase(
        case_name=case_name,
        model_family=model_family,
        input_shape=input_shape,
        source_root=str(source_root),
        onnx_glob=onnx_glob,
        checkpoint=checkpoint,
        representative_data=representative_data,
        deployment_goal=deployment_goal,
        notes=notes,
    )


def save_generated_case(case: GeneratedCase, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(case), indent=2), encoding="utf-8")


def render_generated_case(case: GeneratedCase) -> str:
    return json.dumps(asdict(case), indent=2)
