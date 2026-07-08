import argparse
import math
import os
import os.path as osp

import numpy as np
import onnx
import onnxruntime as rt
from onnx import helper


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


def choose_scale(x, percentile):
    arr = np.abs(x.astype(np.float32).reshape(-1))
    rng = float(np.percentile(arr, percentile))
    if rng < 1e-9:
        return 1.0
    return float(2 ** math.floor(math.log2(127.0 / rng)))


def set_attr(node, name, value):
    kept = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, float(value)))


def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def add_all_node_outputs(model):
    existing = {o.name for o in model.graph.output}
    for node in model.graph.node:
        for out in node.output:
            if out and out not in existing:
                model.graph.output.extend([helper.ValueInfoProto(name=out)])
                existing.add(out)
    return model


def run_float_onnx(float_onnx, input_arr):
    model = onnx.load(float_onnx)
    model = add_all_node_outputs(model)
    providers = rt.get_available_providers()
    sess = rt.InferenceSession(model.SerializeToString(), providers=providers)
    inp = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, {inp: input_arr.astype(np.float32)})
    return dict(zip(out_names, outs))


def patch_model(quant_onnx, float_onnx, input_arr, out_path, percentile):
    ort_outs = run_float_onnx(float_onnx, input_arr)
    model = onnx.load(quant_onnx)
    graph_inputs = {i.name for i in model.graph.input}
    scale_map = {}
    for name in graph_inputs:
        scale_map[name] = choose_scale(input_arr, percentile)

    patched = []
    for node in model.graph.node:
        first_input = next((x for x in node.input if x), "")
        if first_input in scale_map:
            in_scale = scale_map[first_input]
            for attr_name in ("input_scale", "A_scale", "left_scale"):
                if get_attr(node, attr_name) is not None:
                    set_attr(node, attr_name, in_scale)
        if node.op_type == "Resize" and first_input in scale_map:
            set_attr(node, "input_scale", scale_map[first_input])

        if node.output and node.output[0] in ort_outs:
            output_name = node.output[0]
            out_scale = choose_scale(ort_outs[output_name], percentile)
            if node.op_type == "Relu" and first_input in scale_map:
                out_scale = scale_map[first_input]
            if node.op_type == "Add":
                input_scales = [scale_map[x] for x in node.input if x in scale_map]
                if input_scales:
                    out_scale = min(input_scales)
            # Sigmoid/HardSwish confidence is naturally [0, 1], keep 1/128 resolution.
            if node.op_type in ("Sigmoid", "HardSwish"):
                out_scale = max(out_scale, 128.0)
            if get_attr(node, "output_scale") is not None:
                set_attr(node, "output_scale", out_scale)
            scale_map[output_name] = out_scale
            patched.append((node.op_type, output_name, out_scale))

        if node.op_type == "Add":
            for idx, attr_name in ((0, "left_scale"), (1, "right_scale")):
                if idx < len(node.input) and node.input[idx] in scale_map and get_attr(node, attr_name) is not None:
                    set_attr(node, attr_name, scale_map[node.input[idx]])

    os.makedirs(osp.dirname(osp.abspath(out_path)), exist_ok=True)
    onnx.save(model, out_path)
    return scale_map, patched


def build_inputs(features):
    def cat_resize(fd, fe):
        # All currently exported decoder refs already have matching powers of two;
        # nearest resize is only needed for dec5/4/3/head glue.
        if fd.shape[-2:] != fe.shape[-2:]:
            ih, iw = fd.shape[2], fd.shape[3]
            oh, ow = fe.shape[2], fe.shape[3]
            hi = np.minimum((np.arange(oh) * ih) // oh, ih - 1)
            wi = np.minimum((np.arange(ow) * iw) // ow, iw - 1)
            fd = fd[:, :, hi][:, :, :, wi]
        return np.concatenate([fd, fe], axis=1).astype(np.float32)

    dec2_in = features["dec2_in_ref"].astype(np.float32)
    return {
        "fe7": features["fe7"],
        "dec5_in": cat_resize(features["fd6_ref"], features["fe6"]),
        "dec4_in": cat_resize(features["fd5_ref"], features["fe5"]),
        "dec3_in": cat_resize(features["fd4_ref"], features["fe4"]),
        "dec2_chunk0_in": dec2_in[:, :80],
        "dec2_chunk1_in": dec2_in[:, 80:160],
        "dec2_up_ref": features["dec2_up_ref"],
        "dec2_block0_ref": features["dec2_block0_ref"],
        "head_in_ref": features["head_in_ref"],
        "dep_dec0_in_ref": features["dep_dec0_in_ref"],
        "gd_dec0_in_ref": features["gd_dec0_in_ref"],
        "cf_dec0_in_ref": features["cf_dec0_in_ref"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-npz", required=True)
    parser.add_argument("--onnx-root", default="onnx_models")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--percentile", type=float, default=99.9)
    args = parser.parse_args()

    features = np.load(args.feature_npz)
    inputs = build_inputs(features)
    rows = ["model,input_key,input_scale,output_scale"]
    os.makedirs(args.out_root, exist_ok=True)

    for model_name, input_key in MODELS.items():
        quant_onnx = osp.join(args.onnx_root, model_name + ".onnx")
        float_onnx = osp.join(args.onnx_root, model_name, model_name + "_simp.onnx")
        out_path = osp.join(args.out_root, model_name + ".onnx")
        if not osp.exists(float_onnx):
            float_onnx = osp.join(args.onnx_root, model_name, model_name + "_org.onnx")
        scale_map, patched = patch_model(
            quant_onnx,
            float_onnx,
            inputs[input_key].astype(np.float32),
            out_path,
            args.percentile,
        )
        output_scale = patched[-1][2] if patched else 1.0
        input_scale = choose_scale(inputs[input_key], args.percentile)
        rows.append(f"{model_name},{input_key},{input_scale},{output_scale}")
        print(model_name, "input", input_key, "input_scale", input_scale, "output_scale", output_scale)

    summary_path = osp.join(args.out_root, "activation_scales.csv")
    with open(summary_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("WROTE", summary_path)


if __name__ == "__main__":
    main()
