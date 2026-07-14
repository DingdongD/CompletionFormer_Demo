import json
from pathlib import Path
from typing import Dict


def suggest_rule_from_result(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data.get("model") or Path(str(data.get("packer_dir", "unknown"))).name
    compile_status = ""
    board_status = ""
    evidence = str(path)
    if "compile" in data:
        compile_status = data["compile"]["compile"].get("status", "")
        if data.get("board"):
            board_status = data["board"]["parsed"].get("status", "")
    elif "parsed" in data:
        board_status = data["parsed"].get("status", "")
    if board_status == "pass":
        decision = "allow"
        action = "allow this exact packed submodel pattern on RHB"
        risk = "low for this exact shape"
    elif board_status == "fail_timeout":
        decision = "rewrite_or_host"
        action = "split subgraph, disable risky runtime mode, or move pattern to Host"
        risk = "high"
    elif board_status == "fail_accuracy":
        decision = "host"
        action = "run first divergent op on Host or recalibrate/split before retry"
        risk = "high"
    elif compile_status == "fail":
        decision = "rewrite_or_host"
        action = "rewrite unsupported pattern or keep it on Host"
        risk = "medium"
    else:
        decision = "probe_or_host"
        action = "generate smaller probes and update rule with board evidence"
        risk = "medium"
    return {
        "id": f"auto.{model}.{decision}",
        "kind": "operator",
        "pattern": model,
        "compile_status": compile_status or "unknown",
        "board_status": board_status or "unknown",
        "decision": decision,
        "action": action,
        "evidence": evidence,
        "risk": risk,
    }


def render_rule_update(rule: Dict[str, str]) -> str:
    return json.dumps(rule, indent=2)
