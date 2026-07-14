from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ScaleCompensationCandidate:
    name: str
    strategy: str
    module_name: str
    status: str
    notes: str


SUPPORTED_STRATEGIES = {
    "single_conv_io_fold",
    "conv_input_fold_host_output",
    "conv_input_fold_conv_output_fold",
    "conv_input_fold_bn_output_fold",
}


def _quote(value: str) -> str:
    return repr(value)


def _render_get_submodule() -> str:
    return """
def _get_submodule(root, path):
    obj = root
    if not path:
        return obj
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj
""".strip()


def render_scale_compensated_wrapper(spec: Dict[str, object], strategy: str) -> str:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"unsupported scale compensation strategy: {strategy}")

    base_module = str(spec["base_module"])
    base_class = str(spec.get("base_class", "Model"))
    input_scale = float(spec.get("input_scale", 1.0))
    output_scale = float(spec.get("output_scale", 1.0))
    input_conv_path = str(spec.get("input_conv_path", ""))
    output_conv_path = str(spec.get("output_conv_path", ""))
    output_bn_path = str(spec.get("output_bn_path", ""))
    wrapper_doc = str(spec.get("description", "scale compensated wrapper"))

    if not input_conv_path:
        raise ValueError("input_conv_path is required for compiler-safe input scale folding")
    if strategy == "single_conv_io_fold" and not input_conv_path:
        raise ValueError("input_conv_path is required for single Conv input/output folding")
    if strategy == "conv_input_fold_conv_output_fold" and not output_conv_path:
        raise ValueError("output_conv_path is required for conv output folding")
    if strategy == "conv_input_fold_bn_output_fold" and not output_bn_path:
        raise ValueError("output_bn_path is required for BN output folding")

    lines: List[str] = [
        f"from {base_module} import {base_class} as _BaseModel",
        f"from {base_module} import ifmap_sz, input_layouts, op_version, batch_size",
        "import torch.nn as nn",
        "",
        f"INPUT_SCALE = {input_scale:.12g}",
        f"OUTPUT_SCALE = {output_scale:.12g}",
        f"SCALE_COMPENSATION_STRATEGY = {_quote(strategy)}",
        "",
        _render_get_submodule(),
        "",
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        f"        # {wrapper_doc}",
        "        self.base = _BaseModel()",
        f"        input_conv = _get_submodule(self.base, {_quote(input_conv_path)})",
        "        input_conv.weight.data.div_(INPUT_SCALE)",
    ]

    if strategy == "single_conv_io_fold":
        lines.extend(
            [
                "        # Single Conv contract: Conv(x / Si) * So.",
                "        # For y = W*x + b, use W' = W * So / Si and b' = b * So.",
                "        input_conv.weight.data.mul_(OUTPUT_SCALE)",
                "        if input_conv.bias is not None:",
                "            input_conv.bias.data.mul_(OUTPUT_SCALE)",
            ]
        )
    elif strategy == "conv_input_fold_conv_output_fold":
        lines.extend(
            [
                "        if input_conv.bias is not None:",
                "            input_conv.bias.data.div_(INPUT_SCALE)",
                f"        output_conv = _get_submodule(self.base, {_quote(output_conv_path)})",
                "        output_conv.weight.data.mul_(OUTPUT_SCALE)",
                "        if output_conv.bias is not None:",
                "            output_conv.bias.data.mul_(OUTPUT_SCALE)",
            ]
        )
    elif strategy == "conv_input_fold_bn_output_fold":
        lines.extend(
            [
                "        if input_conv.bias is not None:",
                "            input_conv.bias.data.div_(INPUT_SCALE)",
                f"        output_bn = _get_submodule(self.base, {_quote(output_bn_path)})",
                "        output_bn.weight.data.mul_(OUTPUT_SCALE)",
                "        output_bn.bias.data.mul_(OUTPUT_SCALE)",
            ]
        )
    else:
        lines.extend(
            [
                "        if input_conv.bias is not None:",
                "            input_conv.bias.data.div_(INPUT_SCALE)",
                "        # Output scale is intentionally not folded.",
                "        # The generated contract must set this boundary output scale to 1.0,",
                "        # or the downstream consumer must absorb this tensor's native qscale.",
            ]
        )

    lines.extend(
        [
            "        self.eval()",
            "",
            "    def forward(self, *args):",
            "        return self.base(*args)",
            "",
        ]
    )
    return "\n".join(lines)


def generate_scale_compensation_candidates(
    spec: Dict[str, object],
    output_dir: Path,
    strategies: Optional[Iterable[str]] = None,
) -> List[ScaleCompensationCandidate]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(spec["name"])
    requested = list(strategies if strategies is not None else spec.get("strategies", []))
    if not requested:
        requested = [
            "single_conv_io_fold",
            "conv_input_fold_conv_output_fold",
            "conv_input_fold_bn_output_fold",
            "conv_input_fold_host_output",
        ]

    candidates: List[ScaleCompensationCandidate] = []
    for strategy in requested:
        module_name = f"{base_name}_{strategy}"
        path = output_dir / f"{module_name}.py"
        try:
            text = render_scale_compensated_wrapper(spec, strategy)
            path.write_text(text, encoding="utf-8")
            candidates.append(
                ScaleCompensationCandidate(
                    name=base_name,
                    strategy=strategy,
                    module_name=module_name,
                    status="generated",
                    notes=str(path),
                )
            )
        except Exception as exc:
            candidates.append(
                ScaleCompensationCandidate(
                    name=base_name,
                    strategy=strategy,
                    module_name=module_name,
                    status="skipped",
                    notes=str(exc),
                )
            )
    return candidates


def render_candidates_tsv(candidates: Iterable[ScaleCompensationCandidate]) -> str:
    lines = ["name\tstrategy\tmodule_name\tstatus\tnotes"]
    for item in candidates:
        lines.append(f"{item.name}\t{item.strategy}\t{item.module_name}\t{item.status}\t{item.notes}")
    return "\n".join(lines) + "\n"
