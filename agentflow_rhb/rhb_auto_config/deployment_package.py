import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from submodel_exporter import ExportedSubmodel, export_rhb_submodels, render_exported_submodels
from package_contract import build_package_contract, save_package_contract


@dataclass(frozen=True)
class DeploymentPackage:
    package_dir: str
    onnx_path: str
    candidate: str
    policy: str
    manifest_json: str
    schedule_md: str
    rhb_submodels_tsv: str
    host_kernels_md: str
    calibration_plan_json: str
    retraining_plan_md: str
    exported_submodels: List[ExportedSubmodel]


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: object) -> None:
    _write(path, json.dumps(data, indent=2) + "\n")


def _select_candidate(deep_result: Dict, candidate_name: str) -> Dict:
    if not candidate_name:
        candidate_name = deep_result["best_candidate"]
    for item in deep_result["candidates"]:
        if item["name"] == candidate_name:
            return item
    raise ValueError(f"candidate '{candidate_name}' not found")


def _render_schedule(deep_result: Dict, candidate: Dict, deployment_graph: Dict) -> str:
    lines = [
        "# Deployment Schedule",
        "",
        f"- source onnx: `{deep_result['onnx_path']}`",
        f"- policy: `{deep_result.get('policy', '')}`",
        f"- candidate: `{candidate['name']}`",
        f"- objective: {candidate['objective']}",
        f"- estimated latency: {candidate['estimated_latency_ms']} ms",
        f"- Host/RHB boundaries: {candidate['boundary_count']}",
        f"- RHB submodels: {candidate['rhb_submodel_count']}",
        f"- Host regions: {candidate['host_region_count']}",
        "",
        "## Execution Order",
        "",
    ]
    for node in deployment_graph.get("nodes", []):
        lines.extend(
            [
                f"### {node['node_id']}: {node['kind']}",
                "",
                f"- payload: `{node['payload']}`",
                f"- inputs: {node['inputs']}",
                f"- outputs: {node['outputs']}",
                f"- estimated latency: {node['estimated_latency_ms']} ms",
                f"- notes: {node['notes']}",
                "",
            ]
        )
    return "\n".join(lines)


def _render_host_kernels(recipe: List[Dict], candidate: Dict) -> str:
    lines = [
        "# Host Kernels and Rewrite Tasks",
        "",
        "Exact Host kernels run on Host. Approximate hardware-aligned rewrites are candidate RHB replacements and require retraining/reference validation.",
        "",
    ]
    if not recipe:
        lines.append("No Host/rewrite kernels were identified.")
        return "\n".join(lines)
    exact_items = [item for item in recipe if item.get("exact")]
    approx_items = [item for item in recipe if not item.get("exact")]
    if exact_items:
        lines.extend(["## Exact Host / Glue Kernels", ""])
    for item in exact_items:
        exact = "exact" if item.get("exact") else "approx"
        retrain = "requires retraining" if item.get("requires_retraining") else "no retraining"
        lines.extend(
            [
                f"## Node {item['node_index']}: {item['op_type']}",
                "",
                f"- name: `{item['node_name']}`",
                f"- kind: {item['kind']}",
                f"- replacement: {item['replacement']}",
                f"- semantics: {exact}",
                f"- retraining: {retrain}",
                f"- reason: {item['reason']}",
                "",
            ]
        )
    if approx_items:
        lines.extend(["## Approximate Hardware-aligned Rewrite Candidates", ""])
    for item in approx_items:
        lines.extend(
            [
                f"## Node {item['node_index']}: {item['op_type']}",
                "",
                f"- name: `{item['node_name']}`",
                f"- kind: {item['kind']}",
                f"- replacement: {item['replacement']}",
                "- semantics: approximate",
                "- retraining: required",
                f"- reason: {item['reason']}",
                f"- selected candidate: `{candidate['name']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _calibration_plan(deep_result: Dict, candidate: Dict) -> Dict:
    return {
        "candidate": candidate["name"],
        "accuracy_tolerance": {
            "preferred_final_error": "1e-3",
            "max_final_error": "1e-2"
        },
        "sample_counts": [128, 64, 32],
        "required_checks": [
            "per-submodel input activation scale",
            "per-submodel output dequant scale",
            "Host glue dequant/requant scale",
            "final output reference comparison",
            "first divergent tensor localization if final error exceeds tolerance"
        ],
        "inputs_needed": [
            "representative preprocessed input tensors",
            "reference model checkpoint",
            "reference output tensors or dataloader"
        ],
        "source_onnx": deep_result["onnx_path"],
    }


def _render_retraining_plan(recipe: List[Dict], candidate: Dict) -> str:
    approx = [item for item in recipe if item.get("requires_retraining")]
    lines = [
        "# Hardware-aware Retraining Plan",
        "",
        f"- candidate: `{candidate['name']}`",
        f"- approximate rewrite count: {len(approx)}",
        "",
    ]
    if not approx:
        lines.append("No approximate rewrite requires retraining for this candidate. Run reference comparison and board validation.")
        return "\n".join(lines)
    lines.extend(
        [
            "## Required Steps",
            "",
            "1. Replace listed ops in the training graph with hardware-aligned variants.",
            "2. Initialize from the reference checkpoint.",
            "3. Distill from the original reference model using representative data.",
            "4. Validate final output error against the 1e-3 to 1e-2 tolerance.",
            "5. Export ONNX and rerun deep-search, compile/cmodel, and board validation.",
            "",
            "## Approximate Rewrites",
            "",
        ]
    )
    for item in approx:
        lines.append(f"- node {item['node_index']} `{item['op_type']}` -> {item['replacement']}")
    return "\n".join(lines)


def generate_deployment_package(deep_search_json: Path, output_dir: Path, candidate_name: str = "") -> DeploymentPackage:
    deep_result = _read_json(deep_search_json)
    candidate = _select_candidate(deep_result, candidate_name)
    deployment_graph = _read_json(Path(candidate["deployment_graph_json"]))
    recipe = _read_json(Path(deep_result["rewrite_recipe_json"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_deep_search": str(deep_search_json),
        "source_onnx": deep_result["onnx_path"],
        "policy": deep_result.get("policy", ""),
        "candidate": candidate,
        "deployment_graph_json": candidate["deployment_graph_json"],
        "region_plan_json": candidate["region_plan_json"],
        "rewrite_recipe_json": deep_result["rewrite_recipe_json"],
    }
    manifest_path = output_dir / "manifest.json"
    schedule_path = output_dir / "schedule.md"
    host_path = output_dir / "host_kernels.md"
    calibration_path = output_dir / "calibration_plan.json"
    retraining_path = output_dir / "retraining_plan.md"
    rhb_tsv_path = output_dir / "rhb_submodels.tsv"
    contract_json_path = output_dir / "package_contract.json"
    contract_md_path = output_dir / "package_contract.md"

    exported: List[ExportedSubmodel] = []
    if candidate.get("split_multi_output"):
        exported = export_rhb_submodels(Path(candidate["region_plan_json"]), output_dir / "rhb_onnx", split_multi_output=True)
    else:
        exported = export_rhb_submodels(Path(candidate["region_plan_json"]), output_dir / "rhb_onnx", split_multi_output=False)

    _write_json(manifest_path, manifest)
    _write(schedule_path, _render_schedule(deep_result, candidate, deployment_graph) + "\n")
    _write(host_path, _render_host_kernels(recipe, candidate) + "\n")
    _write_json(calibration_path, _calibration_plan(deep_result, candidate))
    _write(retraining_path, _render_retraining_plan(recipe, candidate) + "\n")
    _write(rhb_tsv_path, render_exported_submodels(exported) + "\n")
    save_package_contract(build_package_contract(output_dir), contract_json_path, contract_md_path)

    return DeploymentPackage(
        package_dir=str(output_dir),
        onnx_path=deep_result["onnx_path"],
        candidate=candidate["name"],
        policy=deep_result.get("policy", ""),
        manifest_json=str(manifest_path),
        schedule_md=str(schedule_path),
        rhb_submodels_tsv=str(rhb_tsv_path),
        host_kernels_md=str(host_path),
        calibration_plan_json=str(calibration_path),
        retraining_plan_md=str(retraining_path),
        exported_submodels=exported,
    )


def render_deployment_package(package: DeploymentPackage) -> str:
    lines = [
        "# Deployment Package",
        "",
        f"- package: `{package.package_dir}`",
        f"- source onnx: `{package.onnx_path}`",
        f"- candidate: `{package.candidate}`",
        f"- policy: `{package.policy}`",
        "",
        "## Files",
        "",
        f"- manifest: `{package.manifest_json}`",
        f"- schedule: `{package.schedule_md}`",
        f"- RHB submodels: `{package.rhb_submodels_tsv}`",
        f"- Host kernels: `{package.host_kernels_md}`",
        f"- calibration plan: `{package.calibration_plan_json}`",
        f"- retraining plan: `{package.retraining_plan_md}`",
        "",
        f"exported_submodels: {len(package.exported_submodels)}",
    ]
    return "\n".join(lines)
