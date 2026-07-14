import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from annotator import NodeAnnotation, annotate_graph
from deployment_graph import DeploymentGraph, build_deployment_graph
from graph_importer import OnnxGraphInfo, import_onnx
from regions import RegionPlan, build_region_plan, load_effective_budget
from rewrites import RewriteSuggestion, suggest_rewrites
from rule_db import RuleDB
from policy import DeploymentPolicy, load_deployment_policy


@dataclass(frozen=True)
class SearchCandidate:
    name: str
    objective: str
    allow_approx_rewrites: bool
    split_multi_output: bool
    rhb_submodel_count: int
    host_region_count: int
    boundary_count: int
    multi_output_rhb_regions: int
    approximate_rewrite_count: int
    exact_rewrite_count: int
    packed_weight_bytes_est: int
    estimated_latency_ms: float
    score: float
    risk: str
    required_next_steps: List[str]
    region_plan_json: str
    deployment_graph_json: str


@dataclass(frozen=True)
class DeepSearchResult:
    onnx_path: str
    policy: str
    candidates: List[SearchCandidate]
    best_candidate: str
    rewrite_recipe_json: str


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _risk_from_counts(approx_count: int, multi_output_count: int, host_count: int) -> str:
    if approx_count:
        return "high_requires_retraining"
    if multi_output_count:
        return "medium_requires_split"
    if host_count:
        return "medium_host_glue"
    return "low"


def _score(
    graph: DeploymentGraph,
    plan: RegionPlan,
    approx_count: int,
    multi_output_count: int,
    allow_approx_rewrites: bool,
    policy: DeploymentPolicy,
) -> float:
    rhb_nodes = sum(len(region.nodes) for region in plan.regions if region.target == "RHB")
    host_nodes = sum(len(region.nodes) for region in plan.regions if region.target != "RHB")
    packed = sum(region.packed_weight_bytes_est for region in plan.regions if region.target == "RHB")
    weights = policy.score_weights
    return round(
        rhb_nodes * weights.get("rhb_node", 10.0)
        - host_nodes * weights.get("host_node_penalty", 3.0)
        - graph.boundary_count * weights.get("boundary_penalty", 35.0)
        - graph.rhb_submodel_count * weights.get("rhb_submodel_penalty", 4.0)
        - multi_output_count * weights.get("multi_output_unsplit_penalty", 100.0)
        - approx_count * weights.get("approx_rewrite_penalty", 10.0)
        - (weights.get("approx_strategy_penalty", 8.0) if allow_approx_rewrites else 0.0)
        - (packed / (1024 * 1024)) * weights.get("packed_mb_penalty", 0.25)
        - graph.estimated_latency_ms * weights.get("estimated_latency_penalty", 1.0),
        4,
    )


def _rewrite_recipe(info: OnnxGraphInfo, annotations: List[NodeAnnotation]) -> List[dict]:
    by_index = {item.index: item for item in annotations}
    recipe: List[dict] = []
    for node in info.nodes:
        suggestions = suggest_rewrites(node, by_index[node.index])
        for suggestion in suggestions:
            recipe.append(asdict(suggestion))
    return recipe


def run_deep_search(
    onnx_path: Path,
    rules: RuleDB,
    report_dir: Path,
    effective_budget_bytes: int = 0,
    policy_path: Path = None,
) -> DeepSearchResult:
    info = import_onnx(onnx_path)
    annotations = annotate_graph(info, rules)
    budget = effective_budget_bytes or load_effective_budget()
    policy = load_deployment_policy(policy_path) if policy_path else load_deployment_policy()
    stem = onnx_path.stem

    strategy_defs = [
        ("exact_conservative", "only board-allow exact RHB regions; all risky ops on Host", False, False),
        ("exact_split_outputs", "exact RHB regions with board-runtime single-output split", False, True),
        ("latency_rewrite_search", "latency-first search: allow approved approximate rewrites with retraining/reference validation", True, True),
    ]
    candidates: List[SearchCandidate] = []
    recipe = _rewrite_recipe(info, annotations)
    recipe_path = report_dir / f"rewrite_recipe_{stem}.json"
    _write_json(recipe, recipe_path)

    for name, objective, allow_approx, split_outputs in strategy_defs:
        plan = build_region_plan(info, annotations, budget, allow_approx_rewrites=allow_approx)
        graph = build_deployment_graph(plan)
        approx_count = sum(
            1
            for region in plan.regions
            for suggestion in region.rewrite_suggestions
            if region.target == "RHB" and suggestion.requires_retraining
        )
        exact_count = sum(
            1
            for region in plan.regions
            for suggestion in region.rewrite_suggestions
            if suggestion.exact
        )
        multi_output = sum(1 for region in plan.regions if region.target == "RHB" and len(region.boundary_outputs) != 1)
        packed = sum(region.packed_weight_bytes_est for region in plan.regions if region.target == "RHB")
        required = ["software_quant_diagnostics", "compile", "cmodel", "board", "real_feature_boundary_trace"]
        if split_outputs and multi_output:
            required.append("export_single_output_variants")
        if approx_count:
            required.extend(["model_owned_approx_rewrite", "hardware_aware_retraining", "calibration_32_64_128"])
        elif exact_count:
            required.append("reference_equivalence_check")
        region_plan_path = report_dir / f"deep_region_plan_{stem}_{name}.json"
        deployment_path = report_dir / f"deep_deployment_graph_{stem}_{name}.json"
        _write_json(asdict(plan), region_plan_path)
        _write_json(asdict(graph), deployment_path)
        candidates.append(
            SearchCandidate(
                name=name,
                objective=objective,
                allow_approx_rewrites=allow_approx,
                split_multi_output=split_outputs,
                rhb_submodel_count=graph.rhb_submodel_count,
                host_region_count=graph.host_region_count,
                boundary_count=graph.boundary_count,
                multi_output_rhb_regions=multi_output,
                approximate_rewrite_count=approx_count,
                exact_rewrite_count=exact_count,
                packed_weight_bytes_est=packed,
                estimated_latency_ms=graph.estimated_latency_ms,
                score=_score(graph, plan, approx_count, multi_output if not split_outputs else 0, allow_approx, policy),
                risk=_risk_from_counts(approx_count, multi_output if not split_outputs else 0, graph.host_region_count),
                required_next_steps=required,
                region_plan_json=str(region_plan_path),
                deployment_graph_json=str(deployment_path),
            )
        )
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    return DeepSearchResult(
        onnx_path=str(onnx_path),
        policy=policy.name,
        candidates=candidates,
        best_candidate=candidates[0].name if candidates else "",
        rewrite_recipe_json=str(recipe_path),
    )


def save_deep_search_result(result: DeepSearchResult, path: Path) -> None:
    _write_json(asdict(result), path)


def render_deep_search_result(result: DeepSearchResult) -> str:
    lines = [
        "# Deep Search Result",
        "",
        f"- onnx: `{result.onnx_path}`",
        f"- policy: `{result.policy}`",
        f"- best candidate: `{result.best_candidate}`",
        f"- rewrite recipe: `{result.rewrite_recipe_json}`",
        "",
        "score\tcandidate\trisk\trhb\thost\tboundaries\tmulti_out\tapprox\texact_rewrites\tlatency_ms\tnext_steps",
    ]
    for item in result.candidates:
        lines.append(
            f"{item.score}\t{item.name}\t{item.risk}\t{item.rhb_submodel_count}\t{item.host_region_count}\t"
            f"{item.boundary_count}\t{item.multi_output_rhb_regions}\t{item.approximate_rewrite_count}\t"
            f"{item.exact_rewrite_count}\t{item.estimated_latency_ms}\t{','.join(item.required_next_steps)}"
        )
    return "\n".join(lines)
