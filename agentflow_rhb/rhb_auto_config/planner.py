import json
from pathlib import Path
from typing import List

from schema import CaseSpec
from rule_db import RuleDB


def load_case(path: Path) -> CaseSpec:
    with open(path, "r", encoding="utf-8") as f:
        return CaseSpec.from_dict(json.load(f))


def render_plan(case: CaseSpec, rules: RuleDB) -> str:
    high_risk = rules.risk_rules()
    lines: List[str] = [
        f"# RHB Auto Configuration Plan: {case.case_name}",
        "",
        f"- model family: {case.model_family}",
        f"- input shape: {case.input_shape}",
        f"- source models: {case.source_models_root}",
        "",
        "## Initial RHB Regions",
        "",
    ]
    for idx, name in enumerate(case.accepted_rhb_submodels, 1):
        lines.append(f"{idx}. `{name}`")
    lines.extend(
        [
            "",
            "## Host Glue / CPU Regions",
            "",
        ]
    )
    for item in case.host_ops:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Forbidden / Host-by-default Patterns",
            "",
        ]
    )
    for pattern in case.forbidden_patterns:
        lines.append(f"- {pattern}")
    lines.extend(
        [
            "",
            "## Runtime Requirements",
            "",
            "- `rram_only=false` unless a target-specific probe proves otherwise.",
            "- clear stale `wr_done` before every submodel launch.",
            "- validate CModel first, then board output; board result overrides compile result.",
            "- keep each packed RHB submodel below the effective 8MB budget.",
            "",
            "## Rule-driven Warnings",
            "",
        ]
    )
    for rule in high_risk:
        lines.append(f"- `{rule.id}` [{rule.decision}]: {rule.action}")
    lines.extend(
        [
            "",
            "## Next Automation Hooks",
            "",
            "1. Import ONNX/PyTorch graph and annotate each node with `allow`, `host`, `rewrite`, or `probe`.",
            "2. Build maximal RHB regions from consecutive `allow/rewrite_exact` nodes.",
            "3. Split regions by packed weight budget and known shape/layout constraints.",
            "4. Generate submodel source files and Host scheduler glue.",
            "5. Run compile/CModel, then board validation.",
            "6. Feed failures back into the rule DB and re-plan.",
        ]
    )
    return "\n".join(lines)
