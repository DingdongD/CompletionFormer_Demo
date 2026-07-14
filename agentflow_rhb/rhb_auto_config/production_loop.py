import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ProductionGate:
    name: str
    status: str
    owner: str
    reason: str
    command_hint: str
    promotion_criteria: str


@dataclass(frozen=True)
class ProductionPlan:
    model_name: str
    case_path: str
    objective: str
    semantic_policy: str
    latency_policy: str
    gates: List[ProductionGate]
    required_artifacts: List[str]
    success_criteria: List[str]


def build_production_plan(
    model_name: str,
    case_path: Optional[Path] = None,
    allow_approx: bool = True,
    latency_first: bool = True,
) -> ProductionPlan:
    semantic_policy = (
        "try exact compiler-aligned runtime rewrite first; if exact proof is impossible and allow_approx=true, "
        "generate a retrainable approximate HW model; otherwise keep unsupported fragments on Host"
    )
    latency_policy = (
        "minimize Host/RHB round trips and maximize each board-proven single-output RHB subgraph"
        if latency_first
        else "prioritize accuracy and semantic simplicity before launch count"
    )

    gates = [
        ProductionGate(
            name="source-profile",
            status="required",
            owner="Host",
            reason="detect deform conv, dynamic indexing, stride/downsample, activation, and multi-output risks before export",
            command_hint="cli.py profile-source --source-root <model_source>",
            promotion_criteria="all unsupported ops are assigned to exact rewrite, approximate rewrite, or Host fallback",
        ),
        ProductionGate(
            name="onnx-import-and-max-rhb-search",
            status="required",
            owner="AgentFlow",
            reason="search the largest contiguous RHB subgraph under rule, single-output, and 8MB constraints",
            command_hint="cli.py import-onnx --onnx <model.onnx>; cli.py deep-search --onnx <model.onnx>",
            promotion_criteria="best candidate has explicit RHB/Host schedule and no unclassified nodes",
        ),
        ProductionGate(
            name="exact-rewrite-validation",
            status="required",
            owner="AgentFlow",
            reason="strictly equivalent rewrites can be applied at inference without retraining",
            command_hint="cli.py export-conv-splits ...; cli.py make-split-contract ...",
            promotion_criteria="software reference equivalence passes before compile/cmodel",
        ),
        ProductionGate(
            name="approx-retrain",
            status="required" if allow_approx else "disabled",
            owner="Training",
            reason="GELU->ReLU, ConvTranspose->resize+conv, host-sample+1x1, and other approximate rewrites must be checkpoint-owned",
            command_hint="cli.py remote-train --profile <cspn_or_nlspn_profile> --action plan|submit|status|fetch",
            promotion_criteria="aligned model reaches task metric target before board deployment",
        ),
        ProductionGate(
            name="software-quant-diagnostics",
            status="required",
            owner="AgentFlow",
            reason="catch outlier, heavy-tail, saturation, and bad boundary scales before wasting board cycles",
            command_hint="cli.py quant-diagnostics --npz <calibration_features.npz> --channel-axis 1",
            promotion_criteria="high-risk tensors either get a split/fusion/QAT decision or explicit Host fallback",
        ),
        ProductionGate(
            name="compile-cmodel-pack-board",
            status="required",
            owner="RHB",
            reason="CModel pass is necessary but board output is the final truth",
            command_hint="cli.py deploy-loop --model <submodel> --layout <layout> --run-board",
            promotion_criteria="compile pass, cmodel pass, board All same/pass, no stale-output/timeout",
        ),
        ProductionGate(
            name="real-feature-boundary-trace",
            status="required",
            owner="AgentFlow",
            reason="csim_input can pass while real calibration features expose scale compression or saturation",
            command_hint="runner_dump_boundaries.py --samples 32 --compare board ref",
            promotion_criteria="boundary L1/RMSE/corr and effective scale are stable across val32/calibration32",
        ),
        ProductionGate(
            name="end-to-end-board-visualization",
            status="required",
            owner="App",
            reason="production promotion must show board pred vs ref pred vs GT and latency on real samples",
            command_hint="run_board_pipeline.py --dataset val32 --save-grid",
            promotion_criteria="sample-set metric and app visualization meet the model-specific acceptance target",
        ),
    ]

    required_artifacts = [
        "case spec JSON with input shape, checkpoint, and representative data",
        "current Python source hash and checkpoint hash",
        "fresh ONNX export hash and compile/cmodel logs",
        "packer config with rram_only=false unless specifically proven otherwise",
        "scale contract per Host/RHB boundary",
        "software quant diagnostics JSON/TSV for calibration32/64/128",
        "real-feature board trace and val32 visualization grid",
    ]
    success_criteria = [
        "no stale ONNX/packer artifacts; current export matches current checkpoint",
        "all exact rewrites have software reference equivalence proof",
        "all approximate rewrites are represented in the compiler-aligned model and retrained or QAT-validated",
        "maximal RHB regions are chosen by measured latency, not only by compile success",
        "CSPN/NLSPN runners follow the same boundary scale and board trace discipline as CompletionFormer",
    ]
    return ProductionPlan(
        model_name=model_name,
        case_path=str(case_path or ""),
        objective="production Host/RHB deployment loop for SPN-style depth completion models",
        semantic_policy=semantic_policy,
        latency_policy=latency_policy,
        gates=gates,
        required_artifacts=required_artifacts,
        success_criteria=success_criteria,
    )


def save_production_plan(plan: ProductionPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")


def render_production_plan(plan: ProductionPlan) -> str:
    lines = [
        "# Production Deployment Plan",
        "",
        f"- model: `{plan.model_name}`",
        f"- case: `{plan.case_path}`" if plan.case_path else "- case: not specified",
        f"- objective: {plan.objective}",
        f"- semantic_policy: {plan.semantic_policy}",
        f"- latency_policy: {plan.latency_policy}",
        "",
        "## Gates",
        "",
        "status\tgate\towner\treason\tpromotion_criteria\tcommand_hint",
    ]
    for gate in plan.gates:
        lines.append(
            f"{gate.status}\t{gate.name}\t{gate.owner}\t{gate.reason}\t"
            f"{gate.promotion_criteria}\t{gate.command_hint}"
        )
    lines.extend(["", "## Required Artifacts", ""])
    lines.extend(f"- {item}" for item in plan.required_artifacts)
    lines.extend(["", "## Success Criteria", ""])
    lines.extend(f"- {item}" for item in plan.success_criteria)
    return "\n".join(lines)
