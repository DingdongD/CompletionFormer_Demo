import argparse
import json
import math
import os
import os.path as osp
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as rt
import torch
import torch.nn.functional as F
from onnx import helper
from torch.utils.data import DataLoader


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "CompletionFormer"))


MODELS = {
    "completionformer_test.decoder_tiny_dec6_resize_conv_basicblock_nocbam_4x4_to_8x8_ckpt": "fe7",
    "completionformer_test.decoder_tiny_dec5_resize_conv_basicblock_nocbam_8x8_to_16x16_ckpt": "dec5_in",
    "completionformer_test.decoder_tiny_dec4_resize_conv_basicblock_nocbam_16x16_to_32x32_ckpt": "dec4_in",
    "completionformer_test.decoder_tiny_dec3_resize_conv_basicblock_nocbam_32x32_to_64x64_ckpt": "dec3_in",
    "completionformer_test.decoder_tiny_dec2_resize_upconv_in80_chunk0_64x64_to_128x128_ckpt": "dec2_chunk0_in",
    "completionformer_test.decoder_tiny_dec2_resize_upconv_in80_chunk1_64x64_to_128x128_ckpt": "dec2_chunk1_in",
    "completionformer_test.decoder_tiny_dec2_exact_block_conv0_128x128_ckpt": "dec2_up_ref",
    "completionformer_test.decoder_tiny_dec2_exact_block_conv1_128x128_ckpt": "dec2_block0_ref",
    "completionformer_test.head_tiny_dep_dec1_conv_relu_128x128_ckpt": "head_in_ref",
    "completionformer_test.head_tiny_dep_dec0_conv_relu_128x128_ckpt": "dep_dec0_in_ref",
    "completionformer_test.head_tiny_gd_dec1_conv_relu_128x128_ckpt": "head_in_ref",
    "completionformer_test.head_tiny_gd_dec0_conv_128x128_ckpt": "gd_dec0_in_ref",
    "completionformer_test.head_tiny_cf_dec1_conv_relu_128x128_ckpt": "head_in_ref",
    "completionformer_test.head_tiny_cf_dec0_conv_sigmoid_128x128_ckpt": "cf_dec0_in_ref",
}


def choose_scale_from_range(value):
    value = float(value)
    if value < 1e-9:
        return 1.0
    return float(2 ** math.floor(math.log2(127.0 / value)))


def set_attr(node, name, value):
    kept = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, float(value)))


def get_attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return None


def add_all_node_outputs(model):
    existing = {o.name for o in model.graph.output}
    for node in model.graph.node:
        for out in node.output:
            if out and out not in existing:
                model.graph.output.extend([helper.ValueInfoProto(name=out)])
                existing.add(out)
    return model


def concat_like(fd, fe):
    if fd.shape[-2:] != fe.shape[-2:]:
        fd = F.interpolate(fd, size=fe.shape[-2:], mode="nearest")
    return torch.cat((fd, fe), dim=1)


def collect_sample_inputs(model, rgb, dep):
    b = model.backbone
    with torch.no_grad():
        fe1 = b.conv1(torch.cat((b.conv1_rgb(rgb), b.conv1_dep(dep)), dim=1))
        fe2, fe3, fe4, fe5, fe6, fe7 = b.former(fe1)

        fd6 = b.dec6(fe7)
        fd5 = b.dec5(concat_like(fd6, fe6))
        fd4 = b.dec4(concat_like(fd5, fe5))
        fd3 = b.dec3(concat_like(fd4, fe4))

        dec2_in = torch.cat((fd3, fe3), dim=1)
        dec2_resized = F.interpolate(dec2_in, size=b.dec2.out_hw, mode="bilinear", align_corners=False)
        dec2_up = F.relu(b.dec2.up_conv(dec2_resized))
        dec2_block0 = F.relu(b.dec2.block_conv0(dec2_up))
        dec2_block1 = b.dec2.block_conv1(dec2_block0)
        fd2 = F.relu(dec2_block1 + dec2_up)

        head_in = concat_like(fd2, fe2)
        dep_fd1 = b.dep_dec1(head_in)
        dep_dec0_in = concat_like(dep_fd1, fe1)
        gd_fd1 = b.gd_dec1(head_in)
        gd_dec0_in = concat_like(gd_fd1, fe1)
        cf_fd1 = b.cf_dec1(head_in)
        cf_dec0_in = concat_like(cf_fd1, fe1)

    arrays = {
        "fe7": fe7,
        "dec5_in": concat_like(fd6, fe6),
        "dec4_in": concat_like(fd5, fe5),
        "dec3_in": concat_like(fd4, fe4),
        "dec2_chunk0_in": dec2_in[:, :80],
        "dec2_chunk1_in": dec2_in[:, 80:160],
        "dec2_up_ref": dec2_up,
        "dec2_block0_ref": dec2_block0,
        "head_in_ref": head_in,
        "dep_dec0_in_ref": dep_dec0_in,
        "gd_dec0_in_ref": gd_dec0_in,
        "cf_dec0_in_ref": cf_dec0_in,
    }
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in arrays.items()}


def load_model(ckpt_path):
    from ref_model_hw.completionformer_hw import CompletionFormerHWTiny

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = deepcopy(ckpt["args"])
    args.augment = False
    args.use_fallback_encoder = False
    model = CompletionFormerHWTiny(args).eval()
    model.load_state_dict(ckpt["net"], strict=True)
    return model


def make_loader(dataset_root, batch_size):
    sys.path.insert(0, dataset_root)
    from data.nyu import NYU

    split_candidates = [
        osp.join(dataset_root, name)
        for name in os.listdir(dataset_root)
        if name.endswith("_split.json") or name == "split.json"
    ]
    if not split_candidates:
        raise FileNotFoundError(f"No split json found in {dataset_root}")
    split_json = sorted(split_candidates)[0]
    with open(split_json) as f:
        split_data = json.load(f)
    mode = "train" if split_data.get("train") else "test"

    ds_args = SimpleNamespace(
        data_name="NYU",
        dir_data=osp.join(dataset_root, "nyudepth_hdf5"),
        split_json=split_json,
        nyu_height=128,
        nyu_width=128,
        nyu_crop_height=128,
        nyu_crop_width=128,
        augment=False,
        num_sample=500,
    )
    dataset = NYU(ds_args, mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), len(dataset)


class RangeCollector:
    def __init__(self, percentile):
        self.percentile = percentile
        self.values = {}

    def add(self, name, arr):
        arr = np.abs(arr.astype(np.float32)).reshape(-1)
        if arr.size == 0:
            return
        self.values.setdefault(name, []).append(arr)

    def range(self, name):
        chunks = self.values.get(name, [])
        if not chunks:
            return 0.0
        return float(np.percentile(np.concatenate(chunks), self.percentile))


def make_sessions(onnx_root):
    sessions = {}
    output_names = {}
    for model_name in MODELS:
        float_onnx = osp.join(onnx_root, model_name, model_name + "_simp.onnx")
        if not osp.exists(float_onnx):
            float_onnx = osp.join(onnx_root, model_name, model_name + "_org.onnx")
        model = add_all_node_outputs(onnx.load(float_onnx))
        sess = rt.InferenceSession(model.SerializeToString(), providers=rt.get_available_providers())
        sessions[model_name] = sess
        output_names[model_name] = [o.name for o in sess.get_outputs()]
    return sessions, output_names


def collect_ranges(model, loader, sample_count, onnx_root, percentile):
    sessions, output_names = make_sessions(onnx_root)
    collectors = {name: RangeCollector(percentile) for name in MODELS}
    seen = 0
    for batch in loader:
        rgb = batch["rgb"]
        dep = batch["dep"]
        for i in range(rgb.shape[0]):
            sample_inputs = collect_sample_inputs(model, rgb[i : i + 1], dep[i : i + 1])
            for model_name, input_key in MODELS.items():
                arr = sample_inputs[input_key]
                collectors[model_name].add("__input__", arr)
                sess = sessions[model_name]
                inp_name = sess.get_inputs()[0].name
                outs = sess.run(output_names[model_name], {inp_name: arr})
                for out_name, out_arr in zip(output_names[model_name], outs):
                    collectors[model_name].add(out_name, out_arr)
            seen += 1
            print("CALIB_SAMPLE", seen)
            if seen >= sample_count:
                return collectors, seen
    return collectors, seen


def patch_model(model_name, collector, quant_onnx, out_path):
    model = onnx.load(quant_onnx)
    scale_map = {}
    graph_inputs = {i.name for i in model.graph.input}
    input_scale = choose_scale_from_range(collector.range("__input__"))
    for name in graph_inputs:
        scale_map[name] = input_scale

    rows = []
    for node in model.graph.node:
        first_input = next((x for x in node.input if x), "")
        if first_input in scale_map:
            in_scale = scale_map[first_input]
            for attr_name in ("input_scale", "A_scale", "left_scale"):
                if get_attr(node, attr_name) is not None:
                    set_attr(node, attr_name, in_scale)
        if node.op_type == "Resize" and first_input in scale_map:
            set_attr(node, "input_scale", scale_map[first_input])

        if node.output:
            output_name = node.output[0]
            out_scale = choose_scale_from_range(collector.range(output_name))
            if node.op_type == "Relu" and first_input in scale_map:
                out_scale = scale_map[first_input]
            if node.op_type == "Add":
                input_scales = [scale_map[x] for x in node.input if x in scale_map]
                if input_scales:
                    out_scale = min(input_scales)
            if node.op_type in ("Sigmoid", "HardSwish"):
                out_scale = max(out_scale, 128.0)
            if get_attr(node, "output_scale") is not None:
                set_attr(node, "output_scale", out_scale)
            scale_map[output_name] = out_scale
            rows.append((node.op_type, output_name, out_scale))

        if node.op_type == "Add":
            for idx, attr_name in ((0, "left_scale"), (1, "right_scale")):
                if idx < len(node.input) and node.input[idx] in scale_map and get_attr(node, attr_name) is not None:
                    set_attr(node, attr_name, scale_map[node.input[idx]])

    os.makedirs(osp.dirname(osp.abspath(out_path)), exist_ok=True)
    onnx.save(model, out_path)
    output_scale = rows[-1][2] if rows else 1.0
    return input_scale, output_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=osp.join(ROOT, "CompletionFormer/ref_model_hw/model_00030.pt"))
    parser.add_argument("--dataset-root", default=osp.join(ROOT, "artifacts/nyu_val_representative_32_128x128"))
    parser.add_argument("--onnx-root", default=osp.join(ROOT, "onnx_models"))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_model(args.ckpt)
    loader, dataset_len = make_loader(args.dataset_root, args.batch_size)
    count = min(args.samples, dataset_len)
    collectors, seen = collect_ranges(model, loader, count, args.onnx_root, args.percentile)

    os.makedirs(args.out_root, exist_ok=True)
    summary = ["model,input_key,input_scale,output_scale"]
    for model_name, input_key in MODELS.items():
        quant_onnx = osp.join(args.onnx_root, model_name + ".onnx")
        out_path = osp.join(args.out_root, model_name + ".onnx")
        input_scale, output_scale = patch_model(model_name, collectors[model_name], quant_onnx, out_path)
        print(model_name, input_key, input_scale, output_scale)
        summary.append(f"{model_name},{input_key},{input_scale},{output_scale}")

    summary_path = osp.join(args.out_root, "activation_scales.csv")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary) + "\n")
    meta = {
        "samples": seen,
        "percentile": args.percentile,
        "dataset_root": args.dataset_root,
        "ckpt": args.ckpt,
    }
    with open(osp.join(args.out_root, "calibration_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("WROTE", summary_path)


if __name__ == "__main__":
    main()
