import argparse
import importlib.util
import json
import os.path as osp
import sys
import time

import numpy as np
import torch


def timed(label, fn, stats):
    start = time.perf_counter()
    out = fn()
    elapsed = (time.perf_counter() - start) * 1000.0
    stats.append((label, elapsed))
    print(f"LATENCY {label}: {elapsed:.3f} ms")
    return out


def load_runner(path):
    spec = importlib.util.spec_from_file_location("rhb_completionformer_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(root, ckpt_path):
    sys.path.insert(0, root)
    from CompletionFormer.ref_model_hw.completionformer_hw import CompletionFormerHWTiny

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    args.augment = False
    model = CompletionFormerHWTiny(args).eval()
    model.load_state_dict(ckpt["net"], strict=True)
    return model, args


def load_sample(npz_path, index):
    data = np.load(npz_path)
    rgb = torch.from_numpy(data["rgb"][index : index + 1].astype(np.float32))
    dep = torch.from_numpy(data["dep"][index : index + 1].astype(np.float32))
    gt = data["gt"][index : index + 1].astype(np.float32) if "gt" in data.files else None
    return rgb, dep, gt


def tensor_to_np(x):
    return x.detach().cpu().numpy().astype(np.float32)


def compute_board_cpu_features(model, rgb, dep):
    b = model.backbone
    with torch.no_grad():
        fe1 = b.conv1(torch.cat((b.conv1_rgb(rgb), b.conv1_dep(dep)), dim=1))
        fe2, fe3, fe4, fe5, fe6, fe7 = b.former(fe1)
    return {
        "fe1": tensor_to_np(fe1),
        "fe2": tensor_to_np(fe2),
        "fe3": tensor_to_np(fe3),
        "fe4": tensor_to_np(fe4),
        "fe5": tensor_to_np(fe5),
        "fe6": tensor_to_np(fe6),
        "fe7": tensor_to_np(fe7),
    }


def run_rhb_decoder_heads(r, features, stats, cf_dec0_host_sigmoid=True):
    fd6_raw, fd6_f = r.run_scaled_profiled("dec6", r.DEC6, features["fe7"], stats)
    fd5_in = timed("host::concat_dec5", lambda: r.concat_like(fd6_f, features["fe6"]), stats)
    fd5_raw, fd5_f = r.run_scaled_profiled("dec5", r.DEC5, fd5_in, stats)
    fd4_in = timed("host::concat_dec4", lambda: r.concat_like(fd5_f, features["fe5"]), stats)
    fd4_raw, fd4_f = r.run_scaled_profiled("dec4", r.DEC4, fd4_in, stats)
    fd3_in = timed("host::concat_dec3", lambda: r.concat_like(fd4_f, features["fe4"]), stats)
    fd3_raw, fd3_f = r.run_scaled_profiled("dec3", r.DEC3, fd3_in, stats)

    dec2_trace = {}
    fd2_f = r.dec2_resize_upconv_split(fd3_f, features["fe3"], stats, dec2_trace)

    head_in = timed("host::concat_head_in", lambda: r.concat_like(fd2_f, features["fe2"]), stats)
    dep_fd1_raw, dep_fd1_f = r.run_scaled_profiled("dep_dec1_full", r.DEP_DEC1_FULL, head_in, stats)
    dep_dec0_in = timed("host::concat_dep_dec0", lambda: r.concat_like(dep_fd1_f, features["fe1"]), stats)
    init_depth_raw, init_depth_f = r.run_scaled_profiled("dep_dec0", r.DEP_DEC0, dep_dec0_in, stats)

    gd_fd1_raw, gd_fd1_f = r.run_scaled_profiled("gd_dec1_full", r.GD_DEC1_FULL, head_in, stats)
    gd_dec0_in = timed("host::concat_gd_dec0", lambda: r.concat_like(gd_fd1_f, features["fe1"]), stats)
    guide_raw, guide_f = r.run_scaled_profiled("gd_dec0", r.GD_DEC0, gd_dec0_in, stats)

    cf_fd1_raw, cf_fd1_f = r.run_scaled_profiled("cf_dec1", r.CF_DEC1, head_in, stats)
    cf_dec0_in = timed("host::concat_cf_dec0", lambda: r.concat_like(cf_fd1_f, features["fe1"]), stats)
    cf_dec0_model = r.CF_DEC0_CONV_ONLY if cf_dec0_host_sigmoid else r.CF_DEC0
    confidence_raw, confidence_logits_f = r.run_scaled_profiled("cf_dec0", cf_dec0_model, cf_dec0_in, stats)
    if cf_dec0_host_sigmoid:
        confidence_f = timed("host::cf_dec0_sigmoid", lambda: r.sigmoid_float(confidence_logits_f), stats)
    else:
        confidence_f = confidence_logits_f

    return {
        "fd2_f": fd2_f,
        "init_depth_f": init_depth_f,
        "guide_f": guide_f,
        "confidence_f": confidence_f,
        "confidence_logits_f": confidence_logits_f,
        "raw": {
            "fd6": fd6_raw,
            "fd5": fd5_raw,
            "fd4": fd4_raw,
            "fd3": fd3_raw,
            "dep_fd1": dep_fd1_raw,
            "init_depth": init_depth_raw,
            "gd_fd1": gd_fd1_raw,
            "guide": guide_raw,
            "cf_fd1": cf_fd1_raw,
            "confidence": confidence_raw,
        },
    }


def run_cpu_nlspn(model, rgb, dep, init_depth_f, guide_f, confidence_f):
    pred_init = torch.from_numpy(init_depth_f.astype(np.float32)) + dep
    guide = torch.from_numpy(guide_f.astype(np.float32))
    confidence = torch.from_numpy(confidence_f.astype(np.float32))
    with torch.no_grad():
        if model.prop_layer is not None:
            pred, pred_inter, offset, aff, gamma = model.prop_layer(pred_init, guide, confidence, dep, rgb)
        else:
            pred = pred_init
            pred_inter, offset, aff, gamma = [], None, None, None
        pred = torch.clamp(pred, min=0)
    return {
        "pred": tensor_to_np(pred),
        "pred_init": tensor_to_np(pred_init),
        "guide": guide_f.astype(np.float32),
        "confidence": confidence_f.astype(np.float32),
        "pred_inter_count": np.array([len(pred_inter)], dtype=np.int32),
    }


def category(label):
    if label.startswith("rhb::"):
        return "rhb_run_inference"
    if label.startswith("quant::"):
        return "host_quant"
    if label.startswith("dequant::"):
        return "host_dequant"
    if label.startswith("host::"):
        return "host_glue"
    if label.startswith("cpu::"):
        return "board_cpu_pytorch"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/root/workspace/demo_vp_xj/board_cpu_baseline")
    parser.add_argument("--ckpt", default="/home/root/workspace/demo_vp_xj/board_cpu_baseline/CompletionFormer/ref_model_hw/model_00059.pt")
    parser.add_argument("--source-npz", default="/home/root/workspace/demo_vp_xj/board_cpu_baseline/nyu_val32_source_128x128.npz")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--packer", default="/home/root/workspace/demo_vp_xj/packers/packer_decoder_system_128x128_ckpt00059_traincalib32_convonlycf_rram_false")
    parser.add_argument("--runner", default="/home/root/workspace/demo_vp_xj/packers/rhb_completionformer_decoder_system_runner_128x128_ckpt_fine.py")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-npz", default="/tmp/completionformer_e2e_board_cpu_rhb_outputs.npz")
    parser.add_argument("--save-json", default="/tmp/completionformer_e2e_board_cpu_rhb_latency.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    r = load_runner(args.runner)
    r.ac_driver.set_log_level("info")
    r.ac_driver.init_accelerator()
    r.ac_driver.load_model(args.packer)
    r.load_activation_scales(args.packer)
    r.CLEAR_WR_DONE_BEFORE_RUN = True
    r.init_pl_mmap()

    model, model_args = load_model(args.root, args.ckpt)
    rgb, dep, gt = load_sample(args.source_npz, args.sample_index)
    stats = []

    try:
        features = timed("cpu::stem_former_features", lambda: compute_board_cpu_features(model, rgb, dep), stats)
        rhb_out = run_rhb_decoder_heads(r, features, stats, cf_dec0_host_sigmoid=True)
        host_out = timed(
            "cpu::nlspn_propagation",
            lambda: run_cpu_nlspn(
                model,
                rgb,
                dep,
                rhb_out["init_depth_f"],
                rhb_out["guide_f"],
                rhb_out["confidence_f"],
            ),
            stats,
        )
    finally:
        r.close_pl_mmap()

    total = sum(ms for _, ms in stats)
    by_category = {}
    for label, ms in stats:
        by_category[category(label)] = by_category.get(category(label), 0.0) + ms

    payload = {
        "contract": "End-to-end RHBLite-only: raw RGBD -> board CPU stem/former -> RHB decoder/heads -> board CPU NLSPN",
        "sample_index": args.sample_index,
        "threads": args.threads,
        "torch": torch.__version__,
        "total_ms": total,
        "category_breakdown": [
            {"category": key, "ms": value, "pct": 100.0 * value / total}
            for key, value in sorted(by_category.items(), key=lambda item: item[1], reverse=True)
        ],
        "events": [
            {"label": label, "ms": ms, "category": category(label), "pct": 100.0 * ms / total}
            for label, ms in stats
        ],
        "paths": {
            "ckpt": args.ckpt,
            "source_npz": args.source_npz,
            "packer": args.packer,
            "runner": args.runner,
        },
    }

    np.savez(
        args.save_npz,
        rgb=tensor_to_np(rgb),
        dep=tensor_to_np(dep),
        gt=gt if gt is not None else np.array([], dtype=np.float32),
        board_pred=host_out["pred"],
        board_pred_init=host_out["pred_init"],
        board_guide=host_out["guide"],
        board_confidence=host_out["confidence"],
        fd2_f=rhb_out["fd2_f"],
        init_depth_f=rhb_out["init_depth_f"],
        guide_f=rhb_out["guide_f"],
        confidence_f=rhb_out["confidence_f"],
    )
    with open(args.save_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("E2E_CONTRACT:", payload["contract"])
    print(f"LATENCY e2e_board_cpu_rhb_total: {total:.3f} ms")
    for row in payload["category_breakdown"]:
        print(f"BREAKDOWN {row['category']}: {row['ms']:.3f} ms ({row['pct']:.2f}%)")
    print("SAVED_NPZ:", args.save_npz)
    print("SAVED_JSON:", args.save_json)


if __name__ == "__main__":
    main()
