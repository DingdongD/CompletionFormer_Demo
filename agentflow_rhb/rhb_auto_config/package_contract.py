import csv
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ContractSubmodel:
    region_id: str
    onnx_path: str
    status: str
    outputs: List[str]
    layout: str
    quant: Dict[str, object]
    compile_artifacts: Dict[str, str]
    validation: Dict[str, object]


@dataclass(frozen=True)
class PackageContract:
    package_dir: str
    source_manifest: str
    source_onnx: str
    candidate: str
    policy: str
    inputs: List[Dict[str, object]]
    outputs: List[Dict[str, object]]
    host_glue_before: List[Dict[str, object]]
    host_glue_after: List[Dict[str, object]]
    rhb_submodels: List[ContractSubmodel]
    runtime_requirements: Dict[str, object]
    validation_requirements: List[str]


def _read_json_if_exists(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _read_submodels(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _parse_listish(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


def build_package_contract(package_dir: Path) -> PackageContract:
    manifest_path = package_dir / "manifest.json"
    manifest = _read_json_if_exists(manifest_path)
    calibration = _read_json_if_exists(package_dir / "calibration_plan.json")
    submodels = _read_submodels(package_dir / "rhb_submodels.tsv")
    source_onnx = str(manifest.get("source_onnx") or calibration.get("source_onnx") or "")
    candidate = ""
    candidate_obj = manifest.get("candidate")
    if isinstance(candidate_obj, dict):
        candidate = str(candidate_obj.get("name") or "")
    elif candidate_obj:
        candidate = str(candidate_obj)

    contract_submodels: List[ContractSubmodel] = []
    for row in submodels:
        if not row.get("onnx_path") and not row.get("status"):
            continue
        outputs = []
        raw_outputs = row.get("outputs") or row.get("output_name") or ""
        if raw_outputs:
            outputs = _parse_listish(raw_outputs)
        onnx_path = str(row.get("onnx_path") or "")
        compile_dir = ""
        if onnx_path:
            compile_dir = str(package_dir / "compiled" / Path(onnx_path).stem)
        contract_submodels.append(
            ContractSubmodel(
                region_id=str(row.get("region_id") or row.get("split_id") or ""),
                onnx_path=onnx_path,
                status=str(row.get("status") or "declared"),
                outputs=outputs,
                layout="BWC/BCHW per exported submodel manifest; must be validated by compile command",
                quant={
                    "activation_scale": "calibrate per submodel input/output",
                    "weight_scale": "compiler/packer generated",
                    "host_glue_requant": "required across split/sum/concat boundaries",
                },
                compile_artifacts={
                    "compile_output_dir": compile_dir,
                    "packer_dir": str(package_dir / "packers" / Path(onnx_path).stem) if onnx_path else "",
                },
                validation={
                    "cmodel": "required",
                    "board": "required for accepted deployment",
                    "reference_tensor": "required",
                    "tolerance": calibration.get("accuracy_tolerance", {}),
                },
            )
        )

    return PackageContract(
        package_dir=str(package_dir),
        source_manifest=str(manifest_path),
        source_onnx=source_onnx,
        candidate=candidate,
        policy=str(manifest.get("policy") or ""),
        inputs=[
            {
                "name": "input0",
                "source": "preprocessed representative/runtime tensor",
                "layout": "declared by submodel compile layout",
                "quant": "calibrated per first RHB submodel",
            }
        ],
        outputs=[
            {
                "name": "output0",
                "sink": "final Host/RHB scheduler output",
                "layout": "model reference layout",
                "dequant": "required before final reference comparison",
            }
        ],
        host_glue_before=[
            {
                "kind": "preprocess",
                "description": "load sample, resize/normalize/sparsify as model requires, quantize first RHB input",
            }
        ],
        host_glue_after=[
            {
                "kind": "boundary_glue",
                "description": "concat/add/split-sum/dequant/requant/unsupported activations per schedule and host_kernels.md",
            },
            {
                "kind": "postprocess",
                "description": "dequantize final output and compare/render reference outputs",
            },
        ],
        rhb_submodels=contract_submodels,
        runtime_requirements={
            "rram_only": False,
            "clear_wr_done_before_run": True,
            "single_output_per_board_run": True,
            "board_runner": "deploy.py or local package runner",
        },
        validation_requirements=[
            "compile every RHB submodel",
            "run CModel and compare with exported reference tensors",
            "pack with rram_only=false unless explicitly validated",
            "board-run with wr_done clear before every run",
            "collect latency/counters and stale-output markers",
            "update rule DB from pass/fail evidence",
        ],
    )


def render_package_contract(contract: PackageContract) -> str:
    lines = [
        "# Package Contract",
        "",
        f"- package: `{contract.package_dir}`",
        f"- source ONNX: `{contract.source_onnx}`",
        f"- candidate: `{contract.candidate}`",
        f"- policy: `{contract.policy}`",
        "",
        "## Runtime Requirements",
        "",
    ]
    for key, value in contract.runtime_requirements.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## RHB Submodels", ""])
    if not contract.rhb_submodels:
        lines.append("No RHB submodels declared.")
    for item in contract.rhb_submodels:
        lines.extend(
            [
                f"### {item.region_id}",
                "",
                f"- onnx: `{item.onnx_path}`",
                f"- status: {item.status}",
                f"- outputs: {item.outputs}",
                f"- layout: {item.layout}",
                f"- compile output: `{item.compile_artifacts.get('compile_output_dir', '')}`",
                f"- packer dir: `{item.compile_artifacts.get('packer_dir', '')}`",
                "",
            ]
        )
    lines.extend(["## Validation Requirements", ""])
    lines.extend(f"- {item}" for item in contract.validation_requirements)
    return "\n".join(lines)


def save_package_contract(contract: PackageContract, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(contract), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_package_contract(contract) + "\n", encoding="utf-8")
