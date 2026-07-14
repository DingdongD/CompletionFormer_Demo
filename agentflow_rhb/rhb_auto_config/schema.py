from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULE_DB = FRAMEWORK_ROOT / "rule_db" / "rhb_blackbox_rules.seed.json"
DEFAULT_HARDWARE_CONFIG = FRAMEWORK_ROOT / "configs" / "hardware_config.json"
DEFAULT_DEPLOYMENT_POLICY = FRAMEWORK_ROOT / "configs" / "deployment_policy_latency_first.json"
DEFAULT_REMOTE_TRAINING_CONFIG = FRAMEWORK_ROOT / "configs" / "remote_training_profiles.json"
DEFAULT_REPORT_DIR = FRAMEWORK_ROOT / "reports"


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    pattern: str
    compile_status: str
    board_status: str
    decision: str
    action: str
    evidence: str
    risk: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            pattern=str(data["pattern"]),
            compile_status=str(data["compile_status"]),
            board_status=str(data["board_status"]),
            decision=str(data["decision"]),
            action=str(data["action"]),
            evidence=str(data["evidence"]),
            risk=str(data["risk"]),
        )


@dataclass(frozen=True)
class CaseSpec:
    case_name: str
    model_family: str
    input_shape: List[int]
    source_models_root: str
    accepted_rhb_submodels: List[str]
    host_ops: List[str]
    forbidden_patterns: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseSpec":
        return cls(
            case_name=str(data["case_name"]),
            model_family=str(data["model_family"]),
            input_shape=[int(x) for x in data["input_shape"]],
            source_models_root=str(data["source_models_root"]),
            accepted_rhb_submodels=[str(x) for x in data.get("accepted_rhb_submodels", [])],
            host_ops=[str(x) for x in data.get("host_ops", [])],
            forbidden_patterns=[str(x) for x in data.get("forbidden_patterns", [])],
        )
