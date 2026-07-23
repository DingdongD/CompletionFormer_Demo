import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


RISKY_CALLS = {
    "interpolate": "Host resize or hardware-aligned resize+conv; validate if kept on RHB",
    "softmax": "Host/probe; attention block needs shape-specific validation",
    "gelu": "Approx ReLU/PWL-GELU with retraining or Host",
    "sigmoid": "Host true sigmoid unless exact board evidence exists",
    "tanh": "Host by default",
    "div": "Host gate normalization by default",
    "cat": "Host concat glue unless inside validated submodel",
    "narrow": "Host slice/shift glue",
    "pad": "Host shift/pad glue",
    "abs": "Host gate normalization by default",
    "apply": "custom autograd/function call; inspect callee, keep on Host unless explicitly supported",
    "chunk": "Host split/fan-out glue unless inside a validated single-output submodel",
    "clamp": "Host clip/relu-equivalent boundary unless exact RHB lowering is proven",
    "sum": "Host reduction/normalization by default unless simple validated axis reduction",
}

RISKY_MODULES = {
    "ConvTranspose2d": "Replace with resize+Conv2d hardware-aligned variant or Host resize + RHB Conv",
    "LayerNorm": "Host/probe token-friendly layout",
    "GELU": "Approx activation with retraining or Host",
    "LeakyReLU": "Use ReLU hardware-aligned approximation with retraining, or keep activation on Host",
    "MultiheadAttention": "Split/probe attention; Softmax usually Host",
    "ZeroPad2d": "Host shift/pad glue",
    "Conv3d": "Probe separately; CSPN sum_conv may be Host or exact Conv2d rewrite",
    "ModulatedDeformConvFunction": "Unsupported custom deformable propagation. For strict ref NLSPN semantics keep it on Host; fixed-neighbor propagation is approximate only and requires a retrained checkpoint.",
    "DeformConvFunction": "Unsupported custom deformable convolution; keep on Host or replace with compiler-aligned Conv/shift-sum approximation",
}


@dataclass(frozen=True)
class SourceRisk:
    file: str
    line: int
    symbol: str
    kind: str
    recommendation: str


@dataclass(frozen=True)
class SourceProfile:
    source_root: str
    python_files: int
    risk_counts: Dict[str, int]
    risks: List[SourceRisk]


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def profile_source_tree(root: Path, limit: int = 300) -> SourceProfile:
    risks: List[SourceRisk] = []
    py_files = sorted(root.rglob("*.py"))
    for path in py_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                lower = name.lower()
                if lower in RISKY_CALLS:
                    risks.append(
                        SourceRisk(
                            file=str(path),
                            line=int(getattr(node, "lineno", 0)),
                            symbol=name,
                            kind="call",
                            recommendation=RISKY_CALLS[lower],
                        )
                    )
                if name in RISKY_MODULES:
                    risks.append(
                        SourceRisk(
                            file=str(path),
                            line=int(getattr(node, "lineno", 0)),
                            symbol=name,
                            kind="module",
                            recommendation=RISKY_MODULES[name],
                        )
                    )
    counts: Dict[str, int] = {}
    for risk in risks:
        counts[risk.symbol] = counts.get(risk.symbol, 0) + 1
    return SourceProfile(
        source_root=str(root),
        python_files=len(py_files),
        risk_counts=dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        risks=risks[:limit],
    )


def save_source_profile(profile: SourceProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")


def render_source_profile(profile: SourceProfile) -> str:
    lines = [
        "# Source Profile",
        "",
        f"- source root: `{profile.source_root}`",
        f"- python files: {profile.python_files}",
        "",
        "## Risk Counts",
        "",
    ]
    for name, count in profile.risk_counts.items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Risks", ""])
    for risk in profile.risks:
        lines.append(f"- `{risk.file}:{risk.line}` {risk.kind} `{risk.symbol}`: {risk.recommendation}")
    return "\n".join(lines)
