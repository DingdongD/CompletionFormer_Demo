import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from annotator import annotate_graph
from deployment_graph import build_deployment_graph
from graph_importer import import_onnx
from regions import build_region_plan, load_effective_budget
from rule_db import RuleDB


@dataclass(frozen=True)
class PortfolioItem:
    onnx_path: str
    node_count: int
    rhb_submodel_count: int
    host_region_count: int
    boundary_count: int
    max_rhb_nodes: int
    rhb_weight_est: int
    has_approx_rewrite: bool
    score: float


def score_onnx_portfolio(onnx_paths: List[Path], rules: RuleDB) -> List[PortfolioItem]:
    budget = load_effective_budget()
    items: List[PortfolioItem] = []
    for path in onnx_paths:
        try:
            info = import_onnx(path)
            annotations = annotate_graph(info, rules)
            plan = build_region_plan(info, annotations, budget)
            graph = build_deployment_graph(plan)
            rhb_regions = [region for region in plan.regions if region.target == "RHB"]
            max_rhb_nodes = max((len(region.nodes) for region in rhb_regions), default=0)
            rhb_weight_est = sum(region.packed_weight_bytes_est for region in rhb_regions)
            has_approx = any(
                suggestion.requires_retraining
                for region in plan.regions
                for suggestion in region.rewrite_suggestions
            )
            score = (
                max_rhb_nodes * 10.0
                + graph.rhb_submodel_count * 4.0
                - graph.host_region_count * 2.0
                - graph.boundary_count * 3.0
                - (20.0 if has_approx else 0.0)
            )
            items.append(
                PortfolioItem(
                    onnx_path=str(path),
                    node_count=len(info.nodes),
                    rhb_submodel_count=graph.rhb_submodel_count,
                    host_region_count=graph.host_region_count,
                    boundary_count=graph.boundary_count,
                    max_rhb_nodes=max_rhb_nodes,
                    rhb_weight_est=rhb_weight_est,
                    has_approx_rewrite=has_approx,
                    score=round(score, 4),
                )
            )
        except Exception as exc:
            items.append(
                PortfolioItem(
                    onnx_path=str(path),
                    node_count=0,
                    rhb_submodel_count=0,
                    host_region_count=0,
                    boundary_count=0,
                    max_rhb_nodes=0,
                    rhb_weight_est=0,
                    has_approx_rewrite=False,
                    score=-9999.0,
                )
            )
    return sorted(items, key=lambda item: item.score, reverse=True)


def save_portfolio_json(items: List[PortfolioItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in items], indent=2), encoding="utf-8")


def render_portfolio(items: List[PortfolioItem], limit: int = 80) -> str:
    lines = [
        "score\trhb_submodels\thost_regions\tboundaries\tmax_rhb_nodes\trhb_weight_est\tapprox\tpath"
    ]
    for item in items[:limit]:
        lines.append(
            f"{item.score}\t{item.rhb_submodel_count}\t{item.host_region_count}\t{item.boundary_count}\t"
            f"{item.max_rhb_nodes}\t{item.rhb_weight_est}\t{item.has_approx_rewrite}\t{item.onnx_path}"
        )
    return "\n".join(lines)
