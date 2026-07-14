import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RetryAction:
    priority: int
    action: str
    target: str
    reason: str
    exact: bool
    requires_board: bool


@dataclass(frozen=True)
class RetryPlan:
    source: str
    actions: List[RetryAction]


def plan_retries(localization_json: Path) -> RetryPlan:
    data = json.loads(localization_json.read_text(encoding="utf-8"))
    actions: List[RetryAction] = []
    for finding in data.get("findings", []):
        category = finding.get("category", "")
        if category == "board_timeout":
            actions.extend(
                [
                    RetryAction(1, "runtime_retry", "same_packer", "retry with rram_only=false and clear wr_done", True, True),
                    RetryAction(2, "binary_split", "rhb_region", "isolate first op sequence that triggers status wait timeout", True, True),
                    RetryAction(3, "host_fallback", "first_failing_region", "keep timeout pattern on Host if split still fails", True, True),
                ]
            )
        elif category == "board_accuracy":
            actions.extend(
                [
                    RetryAction(1, "activation_host_split", "Sigmoid/GELU/Tanh/Softmax", "mis-lowered activation is common on board", True, True),
                    RetryAction(2, "scale_audit", "boundary_tensors", "verify dequant/requant scale and split-sum glue", True, False),
                    RetryAction(3, "effective_scale_probe", "rhb_outputs", "estimate real board effective output scale per submodel on representative boundary tensors", True, True),
                    RetryAction(4, "host_head_boundary", "prediction_head", "if the stable feature before the head is accurate but final output is poor, keep the head on Host", True, True),
                    RetryAction(5, "first_divergence_probe", "rhb_region", "compare intermediate outputs and split before first divergent op", True, True),
                ]
            )
        elif category == "board_effective_scale_mismatch":
            actions.extend(
                [
                    RetryAction(1, "effective_scale_probe", "rhb_outputs", "measure robust effective scale while excluding saturated samples", True, True),
                    RetryAction(2, "per_output_scale_override", "host_scheduler", "apply only stable per-output effective scales in Host glue", True, False),
                    RetryAction(3, "host_boundary_shift", "unstable_scaled_region", "move the unstable boundary to Host if ratios are sample-dependent", True, True),
                ]
            )
        elif category == "head_error_amplification":
            actions.extend(
                [
                    RetryAction(1, "host_head_boundary", "prediction_head", "run RHB up to the last stable feature tensor and compute the full prediction head on Host", True, True),
                    RetryAction(2, "head_ablation", "final_conv_vs_full_head", "compare RHB head, Host final Conv, Host full depth head, Host full guidance head", True, False),
                    RetryAction(3, "retrainable_head_rewrite", "prediction_head", "only keep head on RHB after QAT/retraining and validation recover task metrics", False, True),
                ]
            )
        elif category == "layerwise_head_divergence":
            actions.extend(
                [
                    RetryAction(1, "layerwise_probe", "prediction_head_layers", "compare each RHB head layer against Host with the same real boundary input", True, True),
                    RetryAction(2, "split_before_first_bad_layer", "prediction_head", "keep earlier matching layers on RHB and move the first divergent layer onward to Host", True, True),
                    RetryAction(3, "head_full_host_candidate", "prediction_head", "score a full Host-head allocation when the first bad layer is followed by sensitive propagation", True, False),
                ]
            )
        elif category == "channelwise_scale_mismatch":
            actions.extend(
                [
                    RetryAction(1, "channelwise_scale_probe", "multi_channel_output", "fit per-channel scale/bias on representative real outputs and evaluate downstream task metric", True, False),
                    RetryAction(2, "channelwise_requant_or_host", "host_glue", "use channel-wise Host correction only if stable across calibration/validation; otherwise move the producing layer to Host", True, False),
                ]
            )
        elif category in {"compile_fail", "cmodel_fail"}:
            actions.extend(
                [
                    RetryAction(1, "apply_exact_rewrite", "unsupported_op", "try exact rewrite such as Conv+BN fold or Linear->Conv1x1", True, False),
                    RetryAction(2, "host_fallback", "unsupported_op", "keep op on Host and rebuild larger neighboring RHB regions", True, True),
                ]
            )
        elif category == "board_runtime":
            actions.extend(
                [
                    RetryAction(1, "split_multi_output_region", "rhb_region", "board runtime requires exactly one output tensor", True, True),
                    RetryAction(2, "custom_runner_probe", "board_runtime", "verify whether a custom runner can consume this packed model", True, True),
                    RetryAction(3, "host_fanout_boundary", "multi_output_boundary", "keep fan-out on Host and export downstream single-output RHB regions", True, True),
                ]
            )
        elif category == "region_risk":
            actions.append(
                RetryAction(2, "split_at_rewrite_node", str(finding.get("evidence", "")), "remove risky op from RHB island", True, True)
            )
    if not actions:
        actions.append(
            RetryAction(1, "collect_evidence", "logs", "no actionable finding; collect compile/cmodel/board logs", True, False)
        )
    actions = sorted(actions, key=lambda item: item.priority)
    return RetryPlan(source=str(localization_json), actions=actions)


def save_retry_plan(plan: RetryPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")


def render_retry_plan(plan: RetryPlan) -> str:
    lines = ["# Retry Plan", "", f"- source: `{plan.source}`", ""]
    for item in plan.actions:
        lines.append(
            f"{item.priority}. {item.action} -> {item.target} | exact={item.exact} | board={item.requires_board} | {item.reason}"
        )
    return "\n".join(lines)
