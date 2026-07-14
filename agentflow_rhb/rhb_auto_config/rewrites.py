from dataclasses import asdict, dataclass
from typing import List

from annotator import NodeAnnotation
from graph_importer import OnnxNodeInfo


@dataclass(frozen=True)
class RewriteSuggestion:
    node_index: int
    node_name: str
    op_type: str
    kind: str
    replacement: str
    exact: bool
    requires_retraining: bool
    reason: str


def suggest_rewrites(node: OnnxNodeInfo, annotation: NodeAnnotation) -> List[RewriteSuggestion]:
    suggestions: List[RewriteSuggestion] = []
    kernel = node.attrs.get("kernel_shape")
    strides = node.attrs.get("strides")

    if node.op_type == "ConvTranspose":
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="hardware_aligned",
                replacement="Host/RHB bilinear resize + Conv2d",
                exact=False,
                requires_retraining=True,
                reason="ConvTranspose should be replaced by the accepted CompletionFormer upsample+conv variant.",
            )
        )
    if node.op_type == "Resize":
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="host_glue",
                replacement="Host bilinear resize, then pass resized tensor to RHB Conv submodel",
                exact=True,
                requires_retraining=False,
                reason="Board evidence showed F.interpolate/Resize is better treated as Host glue.",
            )
        )
    if node.op_type in {"Slice", "Gather"}:
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="host_glue",
                replacement="Host sample/gather/indexing, then pass the already-sampled tensor to the next RHB Conv submodel",
                exact=True,
                requires_retraining=False,
                reason="CSPN stage2 showed RHB-internal x[:,:,::2,::2] compiles and passes random csim but gives wrong real-feature output; Host sample restored b0 main/shortcut to corr~=0.998.",
            )
        )
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="hardware_aligned",
                replacement="Retrainable hardware-aligned resize/upsample replacement, typically Host resize + RHB Conv or folded Resize+Conv block",
                exact=False,
                requires_retraining=True,
                reason="Latency-first policy allows sampling/upsample approximations when retraining keeps final error within 1e-3 to 1e-2.",
            )
        )
    if node.op_type == "Sigmoid":
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="host_glue",
                replacement="RHB Conv-only head + Host true sigmoid",
                exact=True,
                requires_retraining=False,
                reason="Hardware Sigmoid lowered incorrectly in cf_dec0; Host sigmoid reduced final error.",
            )
        )
    if node.op_type in {"Gelu", "GELU"}:
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="approx_activation",
                replacement="ReLU or calibrated PWL-GELU",
                exact=False,
                requires_retraining=True,
                reason="GELU is not a board-proven primitive in the current RHB rule DB.",
            )
        )
    if node.op_type == "Conv" and kernel in ([8, 8], [4, 4], [2, 2]) and kernel == strides:
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="srconv_rewrite",
                replacement="Host spatial sample/downsample + RHB Conv on the already-sampled tensor, or retrained small-conv approximation",
                exact=False,
                requires_retraining=True,
                reason="PVT srconv kernel=stride patterns produced board timeout/fail in several shapes; keep sample/downsample outside RHB graph.",
            )
        )
    if node.op_type == "Conv" and strides in ([2, 2], [2]) and _output_channels_or_zero(node) >= 32:
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="stride2_conv_rewrite",
                replacement="Exact Host offset/sample gather + RHB Conv1x1 partials, then Host sum+bias; for retrained sample_1x1 blocks, Host performs x[:,:,::2,::2] and RHB sees only the sampled tensor",
                exact=True,
                requires_retraining=False,
                reason="Stride-2 Conv and RHB-internal sample/Slice are not board-safe in CSPN/CompletionFormer probes. Exact offset/sample decomposition is board-safe when chunked; CSPN stage2 requires Host sample before RHB Conv.",
            )
        )
    if node.op_type == "Conv" and _small_spatial_high_channel(node):
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="small_spatial_conv_rewrite",
                replacement="Exact input/output-channel partial Conv; prefer input chunk 8 and output chunk up to 32, then Host sum+bias",
                exact=True,
                requires_retraining=False,
                reason="CSPN down21/down22 board probes showed input chunk 8 passes with output chunk 16/32; input chunk 16+ times out.",
            )
        )
    if node.op_type in {"MatMul", "Gemm"}:
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="linear_layout",
                replacement="Token-friendly Linear probe or Host token op; optionally map Linear to Conv1x1 with padded spatial tile",
                exact=True,
                requires_retraining=False,
                reason="Linear/MatMul support is shape/layout-specific; CompletionFormer used layout-specific workarounds.",
            )
        )
    if node.op_type in {"LayerNorm", "LayerNormalization"}:
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="host_or_probe",
                replacement="Host LayerNorm or token-layout RHB microprobe",
                exact=True,
                requires_retraining=False,
                reason="LayerNorm is layout-sensitive and not globally board-proven.",
            )
        )
    if annotation.decision in {"host", "host_by_default", "rewrite_or_host"} and not suggestions:
        suggestions.append(
            RewriteSuggestion(
                node_index=node.index,
                node_name=node.name,
                op_type=node.op_type,
                kind="host_fallback",
                replacement="Host kernel at CPU/RHB boundary",
                exact=True,
                requires_retraining=False,
                reason=annotation.reason,
            )
        )
    return suggestions


def suggestions_to_dicts(suggestions: List[RewriteSuggestion]) -> List[dict]:
    return [asdict(item) for item in suggestions]


def _output_channels_or_zero(node: OnnxNodeInfo) -> int:
    shape = next(iter(node.output_shapes.values()), []) if node.output_shapes else []
    if len(shape) >= 2:
        try:
            return int(shape[1])
        except Exception:
            return 0
    return 0


def _small_spatial_high_channel(node: OnnxNodeInfo) -> bool:
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
