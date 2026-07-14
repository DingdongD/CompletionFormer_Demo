import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SplitContractStep:
    order: int
    device: str
    kind: str
    name: str
    description: str
    inputs: List[str]
    outputs: List[str]
    constraints: Dict[str, object]


@dataclass(frozen=True)
class SplitContract:
    name: str
    source: str
    conv: Dict[str, object]
    steps: List[SplitContractStep]
    validation: List[str]
    notes: List[str]


def _ceil_chunks(channels: int, chunk: int) -> List[tuple]:
    if channels <= 0:
        raise ValueError("channels must be positive")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    ranges = []
    start = 0
    while start < channels:
        end = min(start + chunk, channels)
        ranges.append((start, end))
        start = end
    return ranges


def build_conv_split_contract(
    name: str,
    source: str,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel: int = 3,
    stride: int = 1,
    padding: int = 1,
    input_chunk: int = 64,
    output_chunk: int = 0,
    post: str = "none",
    pad_output_to: int = 0,
) -> SplitContract:
    """Build a Host/RHB schedule for exact Conv split deployment.

    `post` may be:
      - none: no post-sum affine block
      - bias: RHB 1x1 identity+bias
      - bias_relu: RHB 1x1 identity+bias+ReLU
      - bn_relu: RHB 1x1 folded-BN+ReLU
    """

    if post not in {"none", "bias", "bias_relu", "bn_relu"}:
        raise ValueError(f"unsupported post kind: {post}")
    output_chunk = output_chunk or out_channels
    ic_ranges = _ceil_chunks(in_channels, input_chunk)
    oc_ranges = _ceil_chunks(out_channels, output_chunk)

    steps: List[SplitContractStep] = []
    order = 1
    for oc_start, oc_end in oc_ranges:
        partial_outputs = []
        for ic_start, ic_end in ic_ranges:
            out_name = f"{name}_partial_oc{oc_start}_{oc_end}_ic{ic_start}_{ic_end}"
            partial_outputs.append(out_name)
            steps.append(
                SplitContractStep(
                    order=order,
                    device="RHB",
                    kind="conv_partial",
                    name=out_name,
                    description="Run no-bias/no-activation Conv2d on an input-channel slice.",
                    inputs=[f"input[:,{ic_start}:{ic_end},:,:]"],
                    outputs=[out_name],
                    constraints={
                        "input_channels": ic_end - ic_start,
                        "output_channels": oc_end - oc_start,
                        "kernel": kernel,
                        "stride": stride,
                        "padding": padding,
                        "height": height,
                        "width": width,
                        "single_output": True,
                        "rram_only": False,
                    },
                )
            )
            order += 1
        sum_name = f"{name}_sum_oc{oc_start}_{oc_end}"
        steps.append(
            SplitContractStep(
                order=order,
                device="Host",
                kind="partial_sum",
                name=sum_name,
                description="Sum all input-channel partial outputs before applying bias/BN/ReLU.",
                inputs=partial_outputs,
                outputs=[sum_name],
                constraints={
                    "exact": True,
                    "accumulation": "float32 preferred; int32/dequant/requant requires calibrated scale contract",
                },
            )
        )
        order += 1
        if post != "none":
            post_name = f"{name}_{post}_oc{oc_start}_{oc_end}"
            out_ch = oc_end - oc_start
            rhb_channels = max(out_ch, pad_output_to or out_ch)
            steps.append(
                SplitContractStep(
                    order=order,
                    device="RHB",
                    kind=f"post_sum_{post}",
                    name=post_name,
                    description="Apply post-sum affine as 1x1 Conv, optionally followed by ReLU.",
                    inputs=[sum_name if rhb_channels == out_ch else f"pad_channels({sum_name},{rhb_channels})"],
                    outputs=[post_name],
                    constraints={
                        "input_channels": rhb_channels,
                        "output_channels": rhb_channels,
                        "slice_output_channels": out_ch if rhb_channels != out_ch else 0,
                        "standalone_relu_allowed": False,
                        "one_channel_head_requires_pad8": out_ch == 1,
                    },
                )
            )
            order += 1
    return SplitContract(
        name=name,
        source=source,
        conv={
            "in_channels": in_channels,
            "out_channels": out_channels,
            "height": height,
            "width": width,
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "input_chunk": input_chunk,
            "output_chunk": output_chunk,
            "post": post,
            "pad_output_to": pad_output_to,
        },
        steps=steps,
        validation=[
            "Export each RHB step as a single-output submodel.",
            "Run compile/cmodel for every RHB step.",
            "Run board validation with wr_done clear before every submodel.",
            "Compare Host-assembled output against the unsplit FP32 reference.",
            "If Host sum is quantized, validate dequant/requant scale at the boundary.",
        ],
        notes=[
            "This contract preserves original Conv semantics when partial sums and post-sum affine are applied in order.",
            "Use no-bias/no-activation partial Conv. Bias, BN, and ReLU belong after the full input-channel sum.",
            "For 1-channel post-sum heads, pad to 8 channels on Host, run RHB affine/ReLU, then slice channel 0 on Host.",
        ],
    )


def save_split_contract(contract: SplitContract, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(contract), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_split_contract(contract) + "\n", encoding="utf-8")


def render_split_contract(contract: SplitContract) -> str:
    lines = [
        "# Conv Split Contract",
        "",
        f"- name: `{contract.name}`",
        f"- source: `{contract.source}`",
        f"- conv: `{contract.conv}`",
        "",
        "## Schedule",
        "",
        "order\tdevice\tkind\tname\tinputs\toutputs\tconstraints",
    ]
    for step in contract.steps:
        lines.append(
            f"{step.order}\t{step.device}\t{step.kind}\t{step.name}\t"
            f"{step.inputs}\t{step.outputs}\t{step.constraints}"
        )
    lines.extend(["", "## Validation", ""])
    lines.extend(f"- {item}" for item in contract.validation)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {item}" for item in contract.notes)
    return "\n".join(lines)
