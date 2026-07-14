import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


@dataclass(frozen=True)
class ExportedSubmodel:
    region_id: str
    target: str
    input_names: List[str]
    output_names: List[str]
    onnx_path: str
    status: str
    message: str


def _load_onnx_module():
    try:
        import onnx  # type: ignore
        from onnx import utils as onnx_utils  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Submodel export requires the `onnx` Python package.") from exc
    return onnx, onnx_utils


def _manual_extract_model(source_onnx: str, out_path: Path, region: dict) -> None:
    onnx, _ = _load_onnx_module()
    model = onnx.load(source_onnx)
    graph = model.graph
    node_indices: Set[int] = {int(idx) for idx in region["node_indices"]}
    selected_nodes = [node for idx, node in enumerate(graph.node) if idx in node_indices]
    if not selected_nodes:
        raise RuntimeError("region contains no selected ONNX nodes")

    initializer_by_name = {item.name: item for item in graph.initializer}
    selected_inputs = set()
    selected_outputs = set()
    for node in selected_nodes:
        selected_inputs.update(name for name in node.input if name)
        selected_outputs.update(name for name in node.output if name)

    needed_initializers = [
        initializer_by_name[name]
        for name in sorted(selected_inputs)
        if name in initializer_by_name
    ]

    value_info_by_name: Dict[str, object] = {}
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        value_info_by_name[item.name] = item

    def value_info(name: str):
        if name in value_info_by_name:
            return value_info_by_name[name]
        return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)

    inputs = [value_info(name) for name in region["boundary_inputs"]]
    outputs = [value_info(name) for name in region["boundary_outputs"]]
    subgraph = onnx.helper.make_graph(
        selected_nodes,
        f"{Path(source_onnx).stem}_{region['region_id']}",
        inputs,
        outputs,
        initializer=needed_initializers,
        value_info=[
            item
            for name, item in value_info_by_name.items()
            if name in selected_inputs.union(selected_outputs)
            and name not in set(region["boundary_inputs"])
            and name not in set(region["boundary_outputs"])
        ],
    )
    submodel = onnx.helper.make_model(subgraph, producer_name="rhb_auto_config_manual_extract")
    del submodel.opset_import[:]
    submodel.opset_import.extend(model.opset_import)
    submodel.ir_version = model.ir_version
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(submodel, str(out_path))


def _export_one_region(source_onnx: str, region: dict, output_dir: Path, suffix: str = "") -> ExportedSubmodel:
    _, onnx_utils = _load_onnx_module()
    region_id = f"{region['region_id']}{suffix}"
    input_names = list(region["boundary_inputs"])
    output_names = list(region["boundary_outputs"])
    out_path = output_dir / f"{Path(source_onnx).stem}_{region_id}.onnx"
    if not input_names or not output_names:
        return ExportedSubmodel(
            region_id=region_id,
            target=region["target"],
            input_names=input_names,
            output_names=output_names,
            onnx_path=str(out_path),
            status="skip",
            message="region has empty boundary input or output; manual graph boundary required",
        )
    try:
        onnx_utils.extract_model(source_onnx, str(out_path), input_names, output_names)
        status = "exported"
        message = "ok"
    except Exception as exc:
        try:
            _manual_extract_model(source_onnx, out_path, region)
            status = "exported_manual"
            message = f"onnx.utils.extract_model failed; manual extractor used: {exc}"
        except Exception as manual_exc:
            status = "failed"
            message = f"{exc}; manual extractor also failed: {manual_exc}"
    return ExportedSubmodel(
        region_id=region_id,
        target=region["target"],
        input_names=input_names,
        output_names=output_names,
        onnx_path=str(out_path),
        status=status,
        message=message,
    )


def export_rhb_submodels(
    region_plan_json: Path,
    output_dir: Path,
    split_multi_output: bool = False,
) -> List[ExportedSubmodel]:
    _load_onnx_module()
    with open(region_plan_json, "r", encoding="utf-8") as f:
        plan = json.load(f)
    source_onnx = plan["onnx_path"]
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: List[ExportedSubmodel] = []
    for region in plan["regions"]:
        if region["target"] != "RHB":
            continue
        exported.append(_export_one_region(source_onnx, region, output_dir))
        if split_multi_output and len(region.get("boundary_outputs", [])) > 1:
            for idx, output_name in enumerate(region["boundary_outputs"]):
                split_region = dict(region)
                split_region["boundary_outputs"] = [output_name]
                exported.append(_export_one_region(source_onnx, split_region, output_dir, suffix=f"_out{idx}"))
    return exported


def render_exported_submodels(items: List[ExportedSubmodel]) -> str:
    lines = ["region_id\tstatus\tonnx_path\tinputs\toutputs\tmessage"]
    for item in items:
        lines.append(
            f"{item.region_id}\t{item.status}\t{item.onnx_path}\t{item.input_names}\t{item.output_names}\t{item.message}"
        )
    return "\n".join(lines)
