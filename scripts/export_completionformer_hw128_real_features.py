import argparse
import json
import os
import os.path as osp
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "CompletionFormer"))


def load_model(ckpt_path):
    from ref_model_hw.completionformer_hw import CompletionFormerHWTiny

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = deepcopy(ckpt["args"])
    args.augment = False
    args.use_fallback_encoder = False
    model = CompletionFormerHWTiny(args).eval()
    model.load_state_dict(ckpt["net"], strict=True)
    return model, args


def quantize_int8(x, scale):
    return np.clip(np.rint(x / scale), -128, 127).astype(np.int8)


def concat_like(fd, fe):
    if fd.shape[-2:] != fe.shape[-2:]:
        fd = F.interpolate(fd, size=fe.shape[-2:], mode="nearest")
    return torch.cat((fd, fe), dim=1)


def collect_features_and_ref(model, rgb, dep):
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
        init_depth_raw = b.dep_dec0(dep_dec0_in)
        gd_fd1 = b.gd_dec1(head_in)
        gd_dec0_in = concat_like(gd_fd1, fe1)
        guide = b.gd_dec0(gd_dec0_in)
        cf_fd1 = b.cf_dec1(head_in)
        cf_dec0_in = concat_like(cf_fd1, fe1)
        confidence = b.cf_dec0(cf_dec0_in)

        sample = {"rgb": rgb, "dep": dep}
        out = model(sample)

    return {
        "fe1": fe1,
        "fe2": fe2,
        "fe3": fe3,
        "fe4": fe4,
        "fe5": fe5,
        "fe6": fe6,
        "fe7": fe7,
        "fd6_ref": fd6,
        "fd5_ref": fd5,
        "fd4_ref": fd4,
        "fd3_ref": fd3,
        "dec2_in_ref": dec2_in,
        "dec2_up_ref": dec2_up,
        "dec2_block0_ref": dec2_block0,
        "dec2_block1_ref": dec2_block1,
        "fd2_ref": fd2,
        "head_in_ref": head_in,
        "dep_fd1_ref": dep_fd1,
        "dep_dec0_in_ref": dep_dec0_in,
        "init_depth_raw_ref": init_depth_raw,
        "gd_fd1_ref": gd_fd1,
        "gd_dec0_in_ref": gd_dec0_in,
        "guide_ref": guide,
        "cf_fd1_ref": cf_fd1,
        "cf_dec0_in_ref": cf_dec0_in,
        "confidence_ref": confidence,
        "pred_init_ref": out["pred_init"],
        "pred_ref": out["pred"],
    }


def tensor_to_np(x):
    return x.detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=osp.join(ROOT, "CompletionFormer/ref_model_hw/model_00030.pt"))
    parser.add_argument(
        "--source-npz",
        default=osp.join(ROOT, "artifacts/visualizations/nyu_ref_model_hw_128x128_ckpt/nyu_first4_forward_outputs.npz"),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--feature-scale", type=float, default=1.0)
    args = parser.parse_args()

    model, model_args = load_model(args.ckpt)
    source = np.load(args.source_npz)
    idx = args.sample_index
    rgb = torch.from_numpy(source["rgb"][idx : idx + 1].astype(np.float32))
    dep = torch.from_numpy(source["dep"][idx : idx + 1].astype(np.float32))
    gt = torch.from_numpy(source["gt"][idx : idx + 1].astype(np.float32))

    items = collect_features_and_ref(model, rgb, dep)
    arrays = {
        "rgb": tensor_to_np(rgb),
        "dep": tensor_to_np(dep),
        "gt": tensor_to_np(gt),
        "sample_index": np.array([idx], dtype=np.int32),
        "feature_scale": np.array([args.feature_scale], dtype=np.float32),
    }

    for name, value in items.items():
        arr = tensor_to_np(value)
        arrays[name] = arr
        if name.startswith("fe"):
            arrays[name + "_i8"] = quantize_int8(arr, args.feature_scale)

    if "pred" in source.files:
        arrays["pred_ref_source"] = source["pred"][idx : idx + 1].astype(np.float32)
    if "pred_init" in source.files:
        arrays["pred_init_ref_source"] = source["pred_init"][idx : idx + 1].astype(np.float32)

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **arrays)

    summary = {}
    for key in ("fe1", "fe2", "fe3", "fe4", "fe5", "fe6", "fe7", "pred_init_ref", "pred_ref"):
        arr = arrays[key]
        summary[key] = {
            "shape": list(arr.shape),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
    summary["ckpt"] = args.ckpt
    summary["source_npz"] = args.source_npz
    summary["sample_index"] = idx
    summary["feature_scale"] = args.feature_scale
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    print("EXPORTED_REAL_FEATURES:", args.out)
    print("sample_index:", idx)
    print("feature_scale:", args.feature_scale)
    for key in ("fe7", "fe6", "fe5", "fe4", "fe3", "fe2", "fe1", "pred_ref"):
        arr = arrays[key]
        print(key, arr.shape, arr.dtype, float(arr.min()), float(arr.max()), float(arr.mean()))


if __name__ == "__main__":
    main()
