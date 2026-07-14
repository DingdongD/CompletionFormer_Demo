import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OnnxNodeInfo:
    index: int
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any]
    input_shapes: Dict[str, List[int]]
    output_shapes: Dict[str, List[int]]
    initializer_bytes: int
    packed_weight_bytes_est: int


@dataclass(frozen=True)
class OnnxGraphInfo:
    path: str
    ir_version: int
    producer_name: str
    opset_imports: Dict[str, int]
    inputs: List[str]
    outputs: List[str]
    initializers: List[str]
    initializer_bytes: Dict[str, int]
    nodes: List[OnnxNodeInfo]
    op_histogram: Dict[str, int]


def _load_onnx_module():
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ONNX importer requires the `onnx` Python package. Install it in the host env "
            "or run this command in the existing compiler environment."
        ) from exc
    return onnx


def _attr_to_python(onnx: Any, attr: Any) -> Any:
    value = onnx.helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, bytes):
                out.append(item.decode("utf-8", errors="replace"))
            else:
                out.append(str(item) if item.__class__.__name__.endswith("Proto") else item)
        return out
    if value.__class__.__name__.endswith("Proto"):
        return str(value)
    return value


def _tensor_shape(value_info: Any) -> List[int]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    shape: List[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(-1)
        else:
            shape.append(-1)
    return shape


def _collect_shapes(graph: Any) -> Dict[str, List[int]]:
    shapes: Dict[str, List[int]] = {}
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        shapes[item.name] = _tensor_shape(item)
    return shapes


def _initializer_nbytes(initializer: Any) -> int:
    if initializer.raw_data:
        return len(initializer.raw_data)
    elem_count = 1
    for dim in initializer.dims:
        elem_count *= int(dim)
    dtype_sizes = {
        1: 4,   # FLOAT
        2: 1,   # UINT8
        3: 1,   # INT8
        4: 2,   # UINT16
        5: 2,   # INT16
        6: 4,   # INT32
        7: 8,   # INT64
        10: 2,  # FLOAT16
        11: 8,  # DOUBLE
        12: 4,  # UINT32
        13: 8,  # UINT64
        16: 2,  # BFLOAT16
    }
    return elem_count * dtype_sizes.get(int(initializer.data_type), 4)


def _packed_estimate_from_initializer_bytes(byte_count: int) -> int:
    if byte_count <= 0:
        return 0
    # Most source checkpoints are exported as FP32. Packed INT8 weight is roughly
    # one quarter of FP32 plus compiler metadata/alignment. Keep this conservative.
    return int(byte_count * 0.35) + 256


def import_onnx(path: Path) -> OnnxGraphInfo:
    onnx = _load_onnx_module()
    model = onnx.load(str(path))
    try:
        model_for_shapes = onnx.shape_inference.infer_shapes(model)
    except Exception:
        model_for_shapes = model
    graph = model.graph
    shape_graph = model_for_shapes.graph
    tensor_shapes = _collect_shapes(shape_graph)
    initializer_bytes = {item.name: _initializer_nbytes(item) for item in graph.initializer}
    initializer_names = set(initializer_bytes)
    nodes: List[OnnxNodeInfo] = []
    hist: Dict[str, int] = {}
    for idx, node in enumerate(graph.node):
        attrs = {attr.name: _attr_to_python(onnx, attr) for attr in node.attribute}
        node_initializer_bytes = sum(initializer_bytes.get(name, 0) for name in node.input)
        nodes.append(
            OnnxNodeInfo(
                index=idx,
                name=node.name or f"{node.op_type}_{idx}",
                op_type=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                attrs=attrs,
                input_shapes={name: tensor_shapes.get(name, []) for name in node.input},
                output_shapes={name: tensor_shapes.get(name, []) for name in node.output},
                initializer_bytes=node_initializer_bytes,
                packed_weight_bytes_est=_packed_estimate_from_initializer_bytes(node_initializer_bytes),
            )
        )
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    opsets = {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}
    inputs = [item.name for item in graph.input if item.name not in initializer_names]
    return OnnxGraphInfo(
        path=str(path),
        ir_version=int(model.ir_version),
        producer_name=str(model.producer_name),
        opset_imports=opsets,
        inputs=inputs,
        outputs=[item.name for item in graph.output],
        initializers=sorted(initializer_names),
        initializer_bytes=initializer_bytes,
        nodes=nodes,
        op_histogram=dict(sorted(hist.items())),
    )


def render_onnx_summary(info: OnnxGraphInfo, node_limit: int = 120) -> str:
    lines = [
        f"onnx: {info.path}",
        f"ir_version: {info.ir_version}",
        f"producer: {info.producer_name}",
        f"opsets: {info.opset_imports}",
        f"inputs: {info.inputs}",
        f"outputs: {info.outputs}",
        f"initializers: {len(info.initializers)}",
        f"nodes: {len(info.nodes)}",
        "op_histogram:",
    ]
    lines.extend(f"  {op}: {count}" for op, count in info.op_histogram.items())
    lines.append("")
    lines.append(f"nodes_first_{node_limit}:")
    for node in info.nodes[:node_limit]:
        attr_keys = ",".join(sorted(node.attrs)) if node.attrs else "-"
        out_shapes = list(node.output_shapes.values())
        shape_text = out_shapes[0] if out_shapes else []
        lines.append(
            f"  {node.index:04d} {node.op_type:16s} {node.name:48s} "
            f"packed_est={node.packed_weight_bytes_est} out={shape_text} attrs={attr_keys}"
        )
    return "\n".join(lines)


def save_onnx_json(info: OnnxGraphInfo, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
