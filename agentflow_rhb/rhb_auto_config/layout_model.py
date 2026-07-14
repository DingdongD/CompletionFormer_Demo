import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from graph_importer import OnnxGraphInfo, OnnxNodeInfo


@dataclass(frozen=True)
class TensorLayoutEstimate:
    tensor_name: str
    source_shape: List[int]
    logical_layout: str
    acsim_layout: str
    tile_mode: int
    data_width_bits: int
    padding_shape: List[int]
    padded_shape: List[int]
    element_count: int
    padded_element_count: int
    padding_overhead_ratio: float
    estimated_lbuf_bytes: int
    risk: str
    notes: str


@dataclass(frozen=True)
class NodeLayoutEstimate:
    index: int
    name: str
    op_type: str
    output_tensors: List[TensorLayoutEstimate]
    weight_layout: Dict[str, object]
    risk: str
    risk_reasons: List[str]


@dataclass(frozen=True)
class GraphLayoutReport:
    onnx_path: str
    data_width_bits: int
    nodes: List[NodeLayoutEstimate]
    risk_histogram: Dict[str, int]


def _prod(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        if value <= 0:
            return 0
        out *= int(value)
    return out


def _ceil_to(value: int, granularity: int) -> int:
    if value <= 0:
        return value
    return int(math.ceil(value / granularity) * granularity)


def _risk_rank(risk: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(risk, 1)


def _combine_risk(risks: Sequence[str]) -> str:
    if not risks:
        return "low"
    return max(risks, key=_risk_rank)


def _shape_to_hwc(shape: Sequence[int]) -> tuple:
    # ONNX exports in this project are usually BCHW. Fall back conservatively.
    if len(shape) == 4:
        return int(shape[2]), int(shape[3]), int(shape[1])
    if len(shape) == 3:
        return int(shape[1]), int(shape[2]), int(shape[0])
    if len(shape) == 2:
        return 1, int(shape[1]), int(shape[0])
    if len(shape) == 1:
        return 1, 1, int(shape[0])
    return 0, 0, 0


def _estimate_feature_tensor(name: str, shape: Sequence[int], data_width_bits: int) -> TensorLayoutEstimate:
    h, w, c = _shape_to_hwc(shape)
    tile_mode = 0
    padding_shape = [1, 16, 16] if data_width_bits == 8 else [1, 8, 16]
    padded_h = _ceil_to(h, padding_shape[0])
    padded_w = _ceil_to(w, padding_shape[1])
    padded_c = _ceil_to(c, padding_shape[2])
    elem = _prod([h, w, c])
    padded_elem = _prod([padded_h, padded_w, padded_c])
    overhead = (padded_elem / elem - 1.0) if elem else 0.0
    bytes_est = padded_elem * data_width_bits // 8
    risk = "low"
    notes = "HWC feature map maps to ACSim LBUF tile-mode 0 estimate"
    if elem == 0:
        risk = "medium"
        notes = "unknown or dynamic shape"
    elif overhead > 0.5:
        risk = "medium"
        notes = "large padding overhead"
    if h <= 8 and w <= 8 and c >= 32:
        risk = "high"
        notes = "small-spatial high-channel feature map; known RHB timeout/layout-risk pattern"
    return TensorLayoutEstimate(
        tensor_name=name,
        source_shape=[int(x) for x in shape],
        logical_layout="BCHW" if len(shape) == 4 else "unknown",
        acsim_layout="HWC",
        tile_mode=tile_mode,
        data_width_bits=data_width_bits,
        padding_shape=padding_shape,
        padded_shape=[padded_h, padded_w, padded_c],
        element_count=elem,
        padded_element_count=padded_elem,
        padding_overhead_ratio=round(overhead, 6),
        estimated_lbuf_bytes=bytes_est,
        risk=risk,
        notes=notes,
    )


def _estimate_conv_weight(node: OnnxNodeInfo, data_width_bits: int) -> Dict[str, object]:
    weight_shape: List[int] = []
    for shape in node.input_shapes.values():
        if len(shape) == 4:
            # Initializers can appear in input_shapes when shape inference exposes them.
            # Pick the first 4D non-activation-looking tensor after input0 by name order
            # is unreliable, so this is only an estimate.
            weight_shape = [int(x) for x in shape]
    if not weight_shape:
        return {}
    co, ci, kh, kw = weight_shape
    if data_width_bits == 8:
        padding_shape = [16, 16, 1, 1]
    else:
        padding_shape = [16, 8, 1, 1]
    padded = [
        _ceil_to(co, padding_shape[0]),
        _ceil_to(ci, padding_shape[1]),
        _ceil_to(kh, padding_shape[2]),
        _ceil_to(kw, padding_shape[3]),
    ]
    elem = _prod(weight_shape)
    padded_elem = _prod(padded)
    overhead = (padded_elem / elem - 1.0) if elem else 0.0
    return {
        "acsim_layout": "CoCiKhKw",
        "tile_mode": 0,
        "data_width_bits": data_width_bits,
        "shape": weight_shape,
        "padding_shape": padding_shape,
        "padded_shape": padded,
        "padding_overhead_ratio": round(overhead, 6),
        "estimated_rbuf_bytes": padded_elem * data_width_bits // 8,
    }


def analyze_layout(info: OnnxGraphInfo, data_width_bits: int = 8) -> GraphLayoutReport:
    nodes: List[NodeLayoutEstimate] = []
    hist: Dict[str, int] = {}
    for node in info.nodes:
        outputs = [
            _estimate_feature_tensor(name, shape, data_width_bits)
            for name, shape in node.output_shapes.items()
            if shape
        ]
        reasons: List[str] = []
        risks = [item.risk for item in outputs]
        weight = _estimate_conv_weight(node, data_width_bits) if node.op_type == "Conv" else {}
        if weight and float(weight.get("padding_overhead_ratio", 0.0)) > 0.5:
            risks.append("medium")
            reasons.append("weight padding overhead > 50%")
        if node.op_type == "Conv":
            kernel = node.attrs.get("kernel_shape")
            strides = node.attrs.get("strides")
            out_shape = next(iter(node.output_shapes.values()), [])
            if strides in ([2, 2], [2]):
                risks.append("medium")
                reasons.append("stride-2 Conv should be checked for exact split or sample+1x1 rewrite")
            if kernel in ([8, 8], [4, 4], [2, 2]) and kernel == strides:
                risks.append("high")
                reasons.append("kernel=stride spatial downsample is known rewrite candidate")
            if len(out_shape) == 4:
                c, h, w = int(out_shape[1]), int(out_shape[2]), int(out_shape[3])
                if h <= 8 and w <= 8 and c >= 32:
                    risks.append("high")
                    reasons.append("small-spatial high-channel Conv output")
        for out in outputs:
            if out.risk != "low":
                reasons.append(out.notes)
        risk = _combine_risk(risks)
        hist[risk] = hist.get(risk, 0) + 1
        nodes.append(
            NodeLayoutEstimate(
                index=node.index,
                name=node.name,
                op_type=node.op_type,
                output_tensors=outputs,
                weight_layout=weight,
                risk=risk,
                risk_reasons=sorted(set(reasons)),
            )
        )
    return GraphLayoutReport(
        onnx_path=info.path,
        data_width_bits=data_width_bits,
        nodes=nodes,
        risk_histogram=dict(sorted(hist.items())),
    )


def render_layout_report(report: GraphLayoutReport, limit: int = 160) -> str:
    lines = [
        "# ACSim-style Layout Analysis",
        "",
        f"- onnx: `{report.onnx_path}`",
        f"- data width: {report.data_width_bits} bit",
        f"- risk histogram: {report.risk_histogram}",
        "",
        "index\top_type\trisk\tfirst_output_shape\tpadded_shape\toverhead\test_lbuf_bytes\tname\treasons",
    ]
    for node in report.nodes[:limit]:
        first = node.output_tensors[0] if node.output_tensors else None
        lines.append(
            "\t".join(
                [
                    str(node.index),
                    node.op_type,
                    node.risk,
                    str(first.source_shape if first else []),
                    str(first.padded_shape if first else []),
                    str(first.padding_overhead_ratio if first else ""),
                    str(first.estimated_lbuf_bytes if first else ""),
                    node.name,
                    "; ".join(node.risk_reasons),
                ]
            )
        )
    return "\n".join(lines)


def save_layout_report_json(report: GraphLayoutReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
