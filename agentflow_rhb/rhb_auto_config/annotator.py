import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from graph_importer import OnnxGraphInfo
from rule_db import RuleDB


@dataclass(frozen=True)
class NodeAnnotation:
    index: int
    name: str
    op_type: str
    decision: str
    reason: str
    rule_id: str


DEFAULT_OP_DECISIONS: Dict[str, tuple] = {
    "Conv": ("allow", "Conv is stable when shape/layout is validated", "op.conv3x3.relu.stable"),
    "Relu": ("allow", "ReLU is safe when fused or adjacent to supported Conv", "op.conv3x3.relu.stable"),
    "Add": ("host", "Boundary residual/add glue defaults to Host unless fused in validated submodel", "glue.concat.host"),
    "Concat": ("host", "Concat defaults to Host boundary glue", "glue.concat.host"),
    "Sigmoid": ("host", "Sigmoid must be Host unless exact board-pass exists", "activation.sigmoid.host"),
    "HardSwish": ("host", "HardSwish/Sigmoid lowering needs explicit accuracy validation", "activation.sigmoid.host"),
    "Mul": ("host_by_default", "Tensor gate multiply defaults to Host", "op.tensor_gate_mul.host_by_default"),
    "LayerNormalization": ("probe_or_host", "LayerNorm is layout-sensitive", "op.layernorm.host_or_probe"),
    "LayerNorm": ("probe_or_host", "LayerNorm is layout-sensitive", "op.layernorm.host_or_probe"),
    "Resize": ("rewrite", "Resize+Conv is allowed only as validated hardware-aligned pattern", "rewrite.convtranspose.resize_conv"),
    "ConvTranspose": ("rewrite", "Prefer Resize+Conv hardware-aligned replacement", "rewrite.convtranspose.resize_conv"),
    "MatMul": ("probe_or_host", "MatMul/Linear needs shape-specific board evidence", "op.layernorm.host_or_probe"),
    "Gemm": ("probe_or_host", "Gemm/Linear needs shape-specific board evidence", "op.layernorm.host_or_probe"),
    "Softmax": ("host_by_default", "Softmax requires shape-specific validation", "op.layernorm.host_or_probe"),
    "Div": ("host_by_default", "Tensor division is high-risk glue unless exact board evidence exists", "op.cspn.div.host_by_default"),
    "Abs": ("host_by_default", "Abs is part of CSPN gate normalization and defaults to Host", "op.cspn.gate_norm.host"),
    "Tanh": ("host_by_default", "Tanh is not board-proven in the current rule DB", "activation.tanh.host"),
    "Max": ("host_by_default", "Elementwise max is glue unless fused in a validated submodel", "glue.elementwise.host"),
    "Pad": ("host", "Padding/shift glue defaults to Host boundary", "glue.shift_pad.host"),
    "Slice": ("host", "Slice/downsample/indexing glue must stay on Host; RHB-internal stride sample caused CSPN stage2 failure", "op.slice.downsample.host_sample"),
    "Narrow": ("host", "Narrow/shift glue defaults to Host boundary", "glue.shift_pad.host"),
    "Gather": ("host", "Gather/downsample/indexing glue must stay on Host; feed gathered tensor to RHB Conv submodel", "op.slice.downsample.host_sample"),
    "Squeeze": ("host", "Shape-only glue defaults to Host boundary unless inside validated submodel", "glue.concat.host"),
    "Unsqueeze": ("host", "Shape-only glue defaults to Host boundary unless inside validated submodel", "glue.concat.host"),
    "Reshape": ("host", "Shape-only glue defaults to Host boundary unless inside validated submodel", "glue.concat.host"),
    "Transpose": ("host", "Layout glue defaults to Host boundary unless inside validated submodel", "glue.concat.host"),
}


def _conv_channels_or_zero(node) -> int:
    shape = next(iter(node.output_shapes.values()), []) if node.output_shapes else []
    if len(shape) >= 2:
        try:
            return int(shape[1])
        except Exception:
            return 0
    return 0


def _conv_small_spatial_high_channel(node) -> bool:
    shape = next(iter(node.output_shapes.values()), []) if node.output_shapes else []
    if len(shape) != 4:
        return False
    try:
        channels = int(shape[1])
        height = int(shape[2])
        width = int(shape[3])
    except Exception:
        return False
    return channels >= 32 and height <= 8 and width <= 8


def annotate_graph(info: OnnxGraphInfo, rules: RuleDB) -> List[NodeAnnotation]:
    annotations: List[NodeAnnotation] = []
    for node in info.nodes:
        decision, reason, rule_id = DEFAULT_OP_DECISIONS.get(
            node.op_type,
            ("probe", "Unknown op: generate micro-probe and update rule DB", "unknown.probe"),
        )
        if node.op_type == "Conv":
            kernel = node.attrs.get("kernel_shape")
            strides = node.attrs.get("strides")
            if kernel in ([8, 8], [4, 4], [2, 2]) and kernel == strides:
                decision = "rewrite_or_host"
                reason = "Large-stride srconv-like Conv needs rewrite/host based on CompletionFormer evidence"
                rule_id = "op.srconv.large_stride.rewrite"
            elif strides in ([2, 2], [2]) and _conv_channels_or_zero(node) >= 32:
                decision = "rewrite_or_host"
                reason = "Stride-2/downsample Conv is board-risky; Host must perform sample/gather before RHB Conv or exact offset split"
                rule_id = "op.conv.stride2.high_channel.rewrite"
            elif _conv_small_spatial_high_channel(node):
                decision = "rewrite_or_host"
                reason = "Small-spatial 32+ channel Conv has board timeout/layout risk; keep on Host or use channel-split/1x1 rewrite"
                rule_id = "op.conv.small_spatial.high_channel.rewrite"
        annotations.append(
            NodeAnnotation(
                index=node.index,
                name=node.name,
                op_type=node.op_type,
                decision=decision,
                reason=reason,
                rule_id=rule_id,
            )
        )
    return annotations


def render_annotations(annotations: List[NodeAnnotation]) -> str:
    lines = ["index\top_type\tdecision\trule_id\tname\treason"]
    for item in annotations:
        lines.append(
            f"{item.index}\t{item.op_type}\t{item.decision}\t{item.rule_id}\t{item.name}\t{item.reason}"
        )
    return "\n".join(lines)


def save_annotations_json(annotations: List[NodeAnnotation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in annotations], indent=2), encoding="utf-8")
