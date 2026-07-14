import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from regions import RegionPlan


@dataclass(frozen=True)
class DeploymentNode:
    node_id: str
    kind: str
    payload: str
    inputs: List[str]
    outputs: List[str]
    estimated_latency_ms: float
    notes: List[str]


@dataclass(frozen=True)
class DeploymentGraph:
    source_onnx: str
    nodes: List[DeploymentNode]
    execution_order: List[str]
    rhb_submodel_count: int
    host_region_count: int
    boundary_count: int
    estimated_latency_ms: float


def build_deployment_graph(plan: RegionPlan) -> DeploymentGraph:
    nodes: List[DeploymentNode] = []
    rhb_count = 0
    host_count = 0
    boundary_count = 0
    prev_kind = ""
    total_latency = 0.0
    for region in plan.regions:
        if region.target == "RHB":
            rhb_count += 1
            kind = "RHB_SUBMODEL"
            launch_ms = 3.0
            weight_ms = max(0.2, region.packed_weight_bytes_est / (1024 * 1024) * 0.8)
            compute_ms = max(0.2, len(region.nodes) * 0.15)
            latency = launch_ms + weight_ms + compute_ms
            notes = ["compile+cmodel+board validation required", "rram_only=false", "clear_wr_done_before_run"]
            if len(region.boundary_outputs) != 1:
                notes.append("board runtime requires one output; split this region per output before board-run")
        elif region.target == "HOST_REWRITE":
            host_count += 1
            kind = "HOST_REWRITE"
            latency = max(0.1, len(region.nodes) * 0.25)
            notes = ["apply rewrite/fallback kernel", "validate equivalence; retrain if approximate"]
        else:
            host_count += 1
            kind = "CPU_KERNEL"
            latency = max(0.05, len(region.nodes) * 0.15)
            notes = ["host glue/fallback"]
        if prev_kind and prev_kind != kind:
            boundary_count += 1
            total_latency += 1.0
        prev_kind = kind
        total_latency += latency
        nodes.append(
            DeploymentNode(
                node_id=region.region_id,
                kind=kind,
                payload=",".join(str(idx) for idx in region.node_indices),
                inputs=region.boundary_inputs,
                outputs=region.boundary_outputs,
                estimated_latency_ms=round(latency, 4),
                notes=notes,
            )
        )
    return DeploymentGraph(
        source_onnx=plan.onnx_path,
        nodes=nodes,
        execution_order=[node.node_id for node in nodes],
        rhb_submodel_count=rhb_count,
        host_region_count=host_count,
        boundary_count=boundary_count,
        estimated_latency_ms=round(total_latency, 4),
    )


def save_deployment_graph_json(graph: DeploymentGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(graph), indent=2), encoding="utf-8")


def render_deployment_graph(graph: DeploymentGraph) -> str:
    lines = [
        "# Deployment Graph",
        "",
        f"- source onnx: `{graph.source_onnx}`",
        f"- RHB submodels: {graph.rhb_submodel_count}",
        f"- Host regions: {graph.host_region_count}",
        f"- CPU/RHB boundaries: {graph.boundary_count}",
        f"- rough latency estimate: {graph.estimated_latency_ms} ms",
        "",
        "## Execution Order",
        "",
    ]
    for node in graph.nodes:
        lines.extend(
            [
                f"### {node.node_id}: {node.kind}",
                "",
                f"- payload node indices: `{node.payload}`",
                f"- inputs: {node.inputs}",
                f"- outputs: {node.outputs}",
                f"- estimated latency: {node.estimated_latency_ms} ms",
                f"- notes: {node.notes}",
                "",
            ]
        )
    return "\n".join(lines)
