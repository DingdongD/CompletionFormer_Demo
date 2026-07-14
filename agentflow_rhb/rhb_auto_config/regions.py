import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from annotator import NodeAnnotation
from graph_importer import OnnxGraphInfo, OnnxNodeInfo
from rewrites import RewriteSuggestion, suggest_rewrites, suggestions_to_dicts
from schema import DEFAULT_HARDWARE_CONFIG


RHB_DECISIONS: Set[str] = {"allow", "rewrite_exact"}
HOST_DECISIONS: Set[str] = {"host", "host_by_default", "forbid"}
REWRITE_DECISIONS: Set[str] = {"rewrite", "rewrite_or_host", "probe_or_host", "probe"}


@dataclass(frozen=True)
class RegionNode:
    index: int
    name: str
    op_type: str
    decision: str
    packed_weight_bytes_est: int
    reason: str


@dataclass(frozen=True)
class RegionCandidate:
    region_id: str
    target: str
    node_indices: List[int]
    nodes: List[RegionNode]
    packed_weight_bytes_est: int
    boundary_inputs: List[str]
    boundary_outputs: List[str]
    reason: str
    validation_required: List[str]
    rewrite_suggestions: List[RewriteSuggestion]


@dataclass(frozen=True)
class RegionPlan:
    onnx_path: str
    effective_weight_budget_bytes: int
    regions: List[RegionCandidate]


def load_effective_budget(config_path: Path = DEFAULT_HARDWARE_CONFIG) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("effective_weight_budget_bytes", int(8 * 1024 * 1024 * 0.9)))


def _external_inputs(nodes: Sequence[OnnxNodeInfo], node_set: Set[int], initializer_names: Set[str]) -> List[str]:
    produced_inside = {out for node in nodes if node.index in node_set for out in node.outputs}
    inputs: List[str] = []
    for node in nodes:
        if node.index not in node_set:
            continue
        for name in node.inputs:
            if name and name not in initializer_names and name not in produced_inside and name not in inputs:
                inputs.append(name)
    return inputs


def _external_outputs(nodes: Sequence[OnnxNodeInfo], node_set: Set[int], graph_outputs: Iterable[str]) -> List[str]:
    consumed_outside = {
        name
        for node in nodes
        if node.index not in node_set
        for name in node.inputs
    }
    graph_output_set = set(graph_outputs)
    outputs: List[str] = []
    for node in nodes:
        if node.index not in node_set:
            continue
        for name in node.outputs:
            if name in consumed_outside or name in graph_output_set:
                outputs.append(name)
    return outputs


def _make_region(
    info: OnnxGraphInfo,
    annotations_by_index: Dict[int, NodeAnnotation],
    node_group: List[OnnxNodeInfo],
    region_id: str,
    target: str,
    reason: str,
) -> RegionCandidate:
    node_set = {node.index for node in node_group}
    packed = sum(node.packed_weight_bytes_est for node in node_group)
    region_nodes = [
        RegionNode(
            index=node.index,
            name=node.name,
            op_type=node.op_type,
            decision=annotations_by_index[node.index].decision,
            packed_weight_bytes_est=node.packed_weight_bytes_est,
            reason=annotations_by_index[node.index].reason,
        )
        for node in node_group
    ]
    rewrites: List[RewriteSuggestion] = []
    for node in node_group:
        rewrites.extend(suggest_rewrites(node, annotations_by_index[node.index]))
    validation_required = []
    if target == "RHB":
        validation_required = ["compile", "cmodel", "board", "numeric_compare"]
        if len(_external_outputs(info.nodes, node_set, info.outputs)) != 1:
            validation_required.append("single_output_split_for_board_runtime")
    elif rewrites:
        validation_required = ["host_equivalence", "retraining" if any(not r.exact for r in rewrites) else "numeric_compare"]
    return RegionCandidate(
        region_id=region_id,
        target=target,
        node_indices=[node.index for node in node_group],
        nodes=region_nodes,
        packed_weight_bytes_est=packed,
        boundary_inputs=_external_inputs(info.nodes, node_set, set(info.initializers)),
        boundary_outputs=_external_outputs(info.nodes, node_set, info.outputs),
        reason=reason,
        validation_required=validation_required,
        rewrite_suggestions=rewrites,
    )


def build_region_plan(
    info: OnnxGraphInfo,
    annotations: List[NodeAnnotation],
    effective_weight_budget_bytes: int,
    allow_approx_rewrites: bool = False,
) -> RegionPlan:
    annotations_by_index = {item.index: item for item in annotations}
    regions: List[RegionCandidate] = []
    current_rhb: List[OnnxNodeInfo] = []
    rhb_region_count = 0
    host_region_count = 0

    def flush_rhb() -> None:
        nonlocal current_rhb, rhb_region_count
        if not current_rhb:
            return
        rhb_region_count += 1
        regions.append(
            _make_region(
                info,
                annotations_by_index,
                current_rhb,
                f"rhb_{rhb_region_count:03d}",
                "RHB",
                "maximal contiguous board-eligible region under weight budget",
            )
        )
        current_rhb = []

    for node in info.nodes:
        annotation = annotations_by_index[node.index]
        decision = annotation.decision
        can_rhb = decision in RHB_DECISIONS
        if allow_approx_rewrites and decision == "rewrite":
            can_rhb = True
        if can_rhb and node.op_type == "Relu" and not current_rhb and node.index > 0:
            previous_decision = annotations_by_index[node.index - 1].decision
            if previous_decision not in RHB_DECISIONS:
                can_rhb = False
                decision = "host"
        if can_rhb:
            projected = sum(item.packed_weight_bytes_est for item in current_rhb) + node.packed_weight_bytes_est
            if current_rhb and projected > effective_weight_budget_bytes:
                flush_rhb()
            current_rhb.append(node)
            continue
        flush_rhb()
        host_region_count += 1
        target = "HOST_REWRITE" if decision in REWRITE_DECISIONS else "HOST"
        regions.append(
            _make_region(
                info,
                annotations_by_index,
                [node],
                f"host_{host_region_count:03d}",
                target,
                annotation.reason,
            )
        )
    flush_rhb()
    return RegionPlan(
        onnx_path=info.path,
        effective_weight_budget_bytes=effective_weight_budget_bytes,
        regions=regions,
    )


def save_region_plan_json(plan: RegionPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(plan)
    for region in data["regions"]:
        region["rewrite_suggestions"] = suggestions_to_dicts(
            [RewriteSuggestion(**item) if isinstance(item, dict) else item for item in region["rewrite_suggestions"]]
        )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def render_region_plan(plan: RegionPlan) -> str:
    lines = [
        f"# RHB Region Plan",
        "",
        f"- onnx: `{plan.onnx_path}`",
        f"- effective weight budget: {plan.effective_weight_budget_bytes} bytes",
        f"- regions: {len(plan.regions)}",
        "",
        "## Regions",
        "",
    ]
    for region in plan.regions:
        node_span = f"{region.node_indices[0]}-{region.node_indices[-1]}" if region.node_indices else "-"
        lines.extend(
            [
                f"### {region.region_id}: {region.target}",
                "",
                f"- node span: {node_span}",
                f"- node count: {len(region.nodes)}",
                f"- packed weight est: {region.packed_weight_bytes_est}",
                f"- boundary inputs: {region.boundary_inputs}",
                f"- boundary outputs: {region.boundary_outputs}",
                f"- reason: {region.reason}",
                f"- validation: {region.validation_required}",
                "",
                "nodes:",
            ]
        )
        for node in region.nodes[:80]:
            lines.append(
                f"- {node.index:04d} `{node.op_type}` `{node.name}` decision={node.decision} packed={node.packed_weight_bytes_est}"
            )
        if len(region.nodes) > 80:
            lines.append(f"- ... {len(region.nodes) - 80} more nodes")
        if region.rewrite_suggestions:
            lines.append("")
            lines.append("rewrite/host suggestions:")
            for suggestion in region.rewrite_suggestions:
                exact = "exact" if suggestion.exact else "approx"
                retrain = ", retrain" if suggestion.requires_retraining else ""
                lines.append(
                    f"- node {suggestion.node_index} `{suggestion.op_type}`: {suggestion.replacement} ({exact}{retrain})"
                )
        lines.append("")
    return "\n".join(lines)
