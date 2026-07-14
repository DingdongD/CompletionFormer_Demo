import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from schema import DEFAULT_DEPLOYMENT_POLICY


@dataclass(frozen=True)
class DeploymentPolicy:
    name: str
    objective: List[str]
    accuracy_tolerance: Dict[str, Any]
    calibration: Dict[str, Any]
    runtime_constraints: Dict[str, Any]
    rewrite_policy: Dict[str, str]
    score_weights: Dict[str, float]
    notes: List[str]


def load_deployment_policy(path: Path = DEFAULT_DEPLOYMENT_POLICY) -> DeploymentPolicy:
    data = json.loads(path.read_text(encoding="utf-8"))
    return DeploymentPolicy(
        name=str(data["name"]),
        objective=[str(item) for item in data.get("objective", [])],
        accuracy_tolerance=dict(data.get("accuracy_tolerance", {})),
        calibration=dict(data.get("calibration", {})),
        runtime_constraints=dict(data.get("runtime_constraints", {})),
        rewrite_policy={str(k): str(v) for k, v in data.get("rewrite_policy", {}).items()},
        score_weights={str(k): float(v) for k, v in data.get("score_weights", {}).items()},
        notes=[str(item) for item in data.get("notes", [])],
    )
