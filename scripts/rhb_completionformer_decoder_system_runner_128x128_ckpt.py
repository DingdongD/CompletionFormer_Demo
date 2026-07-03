import argparse
import csv
import mmap
import os
import os.path as osp
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/home/root/workspace/demo_vp_xj")
import ac_driver

ENABLE_DEC6_BARRIER = False
BARRIER_DEC6_FLOAT = None
USE_SECOND_RUN = False
SECOND_RUN_MODE = "none"
CLEAR_WR_DONE_BEFORE_RUN = False
PL_MM = None
PL_FD = None
PL_BASE = 0x800E0000
PL_MAP_SIZE = 0x10000


DEC6 = "completionformer_test.decoder_tiny_dec6_resize_conv_basicblock_nocbam_4x4_to_8x8_ckpt"
DEC5 = "completionformer_test.decoder_tiny_dec5_resize_conv_basicblock_nocbam_8x8_to_16x16_ckpt"
DEC4 = "completionformer_test.decoder_tiny_dec4_resize_conv_basicblock_nocbam_16x16_to_32x32_ckpt"
DEC3 = "completionformer_test.decoder_tiny_dec3_resize_conv_basicblock_nocbam_32x32_to_64x64_ckpt"
DEC2_RSZ_UP0 = "completionformer_test.decoder_tiny_dec2_resize_upconv_in80_chunk0_64x64_to_128x128_ckpt"
DEC2_RSZ_UP1 = "completionformer_test.decoder_tiny_dec2_resize_upconv_in80_chunk1_64x64_to_128x128_ckpt"
DEC2_EXACT_B0 = "completionformer_test.decoder_tiny_dec2_exact_block_conv0_128x128_ckpt"
DEC2_EXACT_B1 = "completionformer_test.decoder_tiny_dec2_exact_block_conv1_128x128_ckpt"

DEP_DEC1_FULL = "completionformer_test.head_tiny_dep_dec1_conv_relu_128x128_ckpt"
DEP_DEC0 = "completionformer_test.head_tiny_dep_dec0_conv_relu_128x128_ckpt"
GD_DEC1_FULL = "completionformer_test.head_tiny_gd_dec1_conv_relu_128x128_ckpt"
GD_DEC0 = "completionformer_test.head_tiny_gd_dec0_conv_128x128_ckpt"
CF_DEC1 = "completionformer_test.head_tiny_cf_dec1_conv_relu_128x128_ckpt"
CF_DEC0 = "completionformer_test.head_tiny_cf_dec0_conv_sigmoid_128x128_ckpt"
CF_DEC0_CONV_ONLY = "completionformer_test.head_tiny_cf_dec0_conv_only_128x128_ckpt"

INPUT_SCALES = {
    DEC6: 256.0,
    DEC5: 128.0,
    DEC4: 64.0,
    DEC3: 16.0,
    DEC2_RSZ_UP0: 32.0,
    DEC2_RSZ_UP1: 32.0,
    DEC2_EXACT_B0: 32.0,
    DEC2_EXACT_B1: 64.0,
    DEP_DEC1_FULL: 16.0,
    DEP_DEC0: 16.0,
    GD_DEC1_FULL: 16.0,
    GD_DEC0: 16.0,
    CF_DEC1: 16.0,
    CF_DEC0: 16.0,
    CF_DEC0_CONV_ONLY: 16.0,
}
OUTPUT_SCALES = {
    DEC6: 128.0,
    DEC5: 32.0,
    DEC4: 16.0,
    DEC3: 32.0,
    DEC2_RSZ_UP0: 16.0,
    DEC2_RSZ_UP1: 8.0,
    DEC2_EXACT_B0: 16.0,
    DEC2_EXACT_B1: 64.0,
    DEP_DEC1_FULL: 8.0,
    DEP_DEC0: 16.0,
    GD_DEC1_FULL: 4.0,
    GD_DEC0: 2.0,
    CF_DEC1: 8.0,
    CF_DEC0: 128.0,
    CF_DEC0_CONV_ONLY: 8.0,
}


def load_activation_scales(model_path):
    path = osp.join(model_path, "activation_scales.csv")
    if not osp.exists(path):
        print("ACTIVATION_SCALES: built-in")
        return
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["model"]
            if name in INPUT_SCALES:
                INPUT_SCALES[name] = float(row["input_scale"])
            if name in OUTPUT_SCALES:
                OUTPUT_SCALES[name] = float(row["output_scale"])
    print("ACTIVATION_SCALES:", path)


def timed(label, fn, stats):
    start = time.perf_counter()
    out = fn()
    elapsed = (time.perf_counter() - start) * 1000.0
    stats.append((label, elapsed))
    print(f"LATENCY {label}: {elapsed:.3f} ms")
    return out


def saturate_int8(x):
    return np.clip(x, -128, 127).astype(np.int8)


def dequant_output(x, output_scale):
    return x.astype(np.float32) / np.float32(output_scale)


def quant_input(x_float, input_scale):
    return saturate_int8(np.rint(x_float.astype(np.float32) * np.float32(input_scale)))


def sigmoid_float(x):
    x = np.clip(x.astype(np.float32), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def run_scaled(sub_model_name, input_float):
    if ENABLE_DEC6_BARRIER and sub_model_name != DEC6 and BARRIER_DEC6_FLOAT is not None:
        barrier_feed = quant_input(BARRIER_DEC6_FLOAT, INPUT_SCALES[DEC6])
        ac_driver.run_inference([barrier_feed], sub_model_name=DEC6)
    feed = quant_input(input_float, INPUT_SCALES[sub_model_name])
    second_models = {
        DEC4,
        DEC3,
        DEC2_RSZ_UP0,
        DEC2_RSZ_UP1,
        DEC2_EXACT_B0,
        DEC2_EXACT_B1,
        DEP_DEC1_FULL,
        DEP_DEC0,
        GD_DEC1_FULL,
        GD_DEC0,
        CF_DEC1,
        CF_DEC0,
        CF_DEC0_CONV_ONLY,
    }
    use_second = USE_SECOND_RUN or (SECOND_RUN_MODE == "from-dec4" and sub_model_name in second_models)
    if CLEAR_WR_DONE_BEFORE_RUN:
        clear_wr_done()
    if use_second:
        ac_driver.run_inference([feed], sub_model_name=sub_model_name)
        if CLEAR_WR_DONE_BEFORE_RUN:
            clear_wr_done()
    raw = ac_driver.run_inference([feed], sub_model_name=sub_model_name).astype(np.int8)
    y_float = dequant_output(raw, OUTPUT_SCALES[sub_model_name])
    return raw, y_float


def init_pl_mmap():
    global PL_FD, PL_MM
    if PL_MM is not None:
        return
    PL_FD = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    PL_MM = mmap.mmap(PL_FD, PL_MAP_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=PL_BASE)


def close_pl_mmap():
    global PL_FD, PL_MM
    if PL_MM is not None:
        PL_MM.close()
        PL_MM = None
    if PL_FD is not None:
        os.close(PL_FD)
        PL_FD = None


def clear_wr_done():
    if PL_MM is None:
        init_pl_mmap()
    # PL offset 0x00 bit8 is the clear/strobe for PL 0x20 bit0 wr_done.
    struct.pack_into("<I", PL_MM, 0x00, 0x100)
    _ = struct.unpack_from("<I", PL_MM, 0x00)[0]


def load_input(model_path, sub_model_name, index=0):
    data = np.load(osp.join(model_path, "csim_input.npz"))
    return data[f"{sub_model_name}.{index}"].astype(np.int8)


def make_feature(shape, salt):
    total = int(np.prod(shape))
    values = (np.arange(total, dtype=np.int32) * 17 + salt) % 255 - 128
    return values.reshape(shape).astype(np.int8)


def load_real_features(path):
    data = np.load(path)
    features = {}
    for name in ("fe1", "fe2", "fe3", "fe4", "fe5", "fe6", "fe7"):
        if name in data.files:
            features[name] = data[name].astype(np.float32)
            continue
        key = name + "_i8"
        if key not in data.files:
            raise ValueError(f"Missing {name}/{key} in feature npz: {path}")
        features[name] = data[key].astype(np.float32)
    return features


def resize_nearest_nchw(x, out_h, out_w):
    in_h, in_w = x.shape[2], x.shape[3]
    h_idx = np.minimum((np.arange(out_h) * in_h) // out_h, in_h - 1)
    w_idx = np.minimum((np.arange(out_w) * in_w) // out_w, in_w - 1)
    return x[:, :, h_idx][:, :, :, w_idx].astype(x.dtype)


def concat_like(fd, fe):
    resized = resize_nearest_nchw(fd, fe.shape[2], fe.shape[3])
    return np.concatenate([resized, fe], axis=1).astype(np.float32)


def dec2_resize_upconv_split(fd3, fe3, stats, trace=None, warmup_up0=False, warmup_all=False, dec2_mode="normal"):
    x64 = np.concatenate([fd3, fe3], axis=1).astype(np.float32)

    def run_upconv():
        if dec2_mode in ("dummy-y1", "dummy-y1-warm-y1"):
            warm_raw, warm_f = run_scaled(DEC2_RSZ_UP1, x64[:, 80:160])
            if trace is not None:
                trace["dec2_dummy_y1_raw"] = warm_raw
                trace["dec2_dummy_y1_f"] = warm_f
        if warmup_up0 or warmup_all:
            warm_raw, warm_f = run_scaled(DEC2_RSZ_UP0, x64[:, :80])
            if trace is not None:
                trace["dec2_warmup_y0_raw"] = warm_raw
                trace["dec2_warmup_y0_f"] = warm_f
        if dec2_mode == "reverse":
            y1_raw, y1_f = run_scaled(DEC2_RSZ_UP1, x64[:, 80:160])
            y0_raw, y0_f = run_scaled(DEC2_RSZ_UP0, x64[:, :80])
        else:
            y0_raw, y0_f = run_scaled(DEC2_RSZ_UP0, x64[:, :80])
            if warmup_all or dec2_mode == "dummy-y1-warm-y1":
                warm_raw, warm_f = run_scaled(DEC2_RSZ_UP1, x64[:, 80:160])
                if trace is not None:
                    trace["dec2_warmup_y1_raw"] = warm_raw
                    trace["dec2_warmup_y1_f"] = warm_f
            y1_raw, y1_f = run_scaled(DEC2_RSZ_UP1, x64[:, 80:160])
        if warmup_all and dec2_mode == "reverse":
            warm_raw, warm_f = run_scaled(DEC2_RSZ_UP1, x64[:, 80:160])
            if trace is not None:
                trace["dec2_warmup_y1_raw"] = warm_raw
                trace["dec2_warmup_y1_f"] = warm_f
        up_f = np.maximum(y0_f + y1_f, 0.0).astype(np.float32)
        if trace is not None:
            trace["dec2_x64"] = x64
            trace["dec2_y0_raw"] = y0_raw
            trace["dec2_y1_raw"] = y1_raw
            trace["dec2_y0_f"] = y0_f
            trace["dec2_y1_f"] = y1_f
            trace["dec2_up_f"] = up_f
        return up_f

    up = timed("dec2_resize_upconv_split_sum_host_relu", run_upconv, stats)
    if warmup_all:
        warm_raw, warm_f = run_scaled(DEC2_EXACT_B0, up)
        if trace is not None:
            trace["dec2_warmup_b0_raw"] = warm_raw
            trace["dec2_warmup_b0_f"] = warm_f
    b0_raw, b0_f = timed("dec2_exact_block_conv0_rhb", lambda: run_scaled(DEC2_EXACT_B0, up), stats)
    if warmup_all:
        warm_raw, warm_f = run_scaled(DEC2_EXACT_B1, b0_f)
        if trace is not None:
            trace["dec2_warmup_b1_raw"] = warm_raw
            trace["dec2_warmup_b1_f"] = warm_f
    b1_raw, b1_f = timed("dec2_exact_block_conv1_rhb", lambda: run_scaled(DEC2_EXACT_B1, b0_f), stats)
    fd2_f = timed("host_dec2_exact_residual_relu", lambda: np.maximum(b1_f + up, 0.0).astype(np.float32), stats)
    if trace is not None:
        trace["dec2_b0_raw"] = b0_raw
        trace["dec2_b1_raw"] = b1_raw
        trace["dec2_b0_f"] = b0_f
        trace["dec2_b1_f"] = b1_f
        trace["dec2_fd2_f"] = fd2_f
    return fd2_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--save-npz", default="")
    parser.add_argument("--feature-npz", default="", help="Optional real feature npz exported by export_completionformer_hw128_real_features.py")
    parser.add_argument("--reset-before-dec2", action="store_true", help="Re-initialize and reload the packer before running dec2.")
    parser.add_argument("--warmup-dec2-up0", action="store_true", help="Run dec2 upconv chunk0 once and discard it before the real chunk0 run.")
    parser.add_argument("--warmup-dec2-all", action="store_true", help="Run each dec2 RHB submodel once and discard it before the real run.")
    parser.add_argument("--dec2-mode", choices=["normal", "reverse", "dummy-y1", "dummy-y1-warm-y1"], default="normal")
    parser.add_argument("--pre-dec2-dummy", choices=["none", "dec6", "dec5", "dec4", "dec3"], default="none")
    parser.add_argument("--dec6-barrier-before-each", action="store_true", help="Run dec6 once and discard it before each non-dec6 submodel call.")
    parser.add_argument("--warmup-head-dec0", action="store_true", help="Run dep/gd/cf dec0 once and discard before using the real result.")
    parser.add_argument("--warmup-head-all", action="store_true", help="Run dep/gd/cf dec1 and dec0 once and discard before using the real result.")
    parser.add_argument("--use-second-run", action="store_true", help="Run every submodel twice with the same input and use the second output.")
    parser.add_argument("--second-run-mode", choices=["none", "from-dec4"], default="none")
    parser.add_argument("--clear-wr-done-before-run", action="store_true", help="Clear stale PL wr_done with PL[0x00]=0x100 before every run_inference.")
    parser.add_argument("--cf-dec0-host-sigmoid", action="store_true", help="Run cf_dec0 Conv on RHB and apply sigmoid on Host.")
    args = parser.parse_args()

    ac_driver.set_log_level("info")
    ac_driver.init_accelerator()
    ac_driver.load_model(args.model_path)
    load_activation_scales(args.model_path)
    stats = []

    needed = [
        DEC6, DEC5, DEC4, DEC3,
        DEC2_RSZ_UP0, DEC2_RSZ_UP1, DEC2_EXACT_B0, DEC2_EXACT_B1,
        DEP_DEC1_FULL, DEP_DEC0, GD_DEC1_FULL, GD_DEC0, CF_DEC1,
    ]
    needed.append(CF_DEC0_CONV_ONLY if args.cf_dec0_host_sigmoid else CF_DEC0)
    available = set(ac_driver.get_available_sub_models())
    missing = [name for name in needed if name not in available]
    if missing:
        raise ValueError(f"Missing sub-models: {missing}")

    if args.feature_npz:
        features = load_real_features(args.feature_npz)
        fe7 = features["fe7"]
        fe6 = features["fe6"]
        fe5 = features["fe5"]
        fe4 = features["fe4"]
        fe3 = features["fe3"]
        fe2 = features["fe2"]
        fe1 = features["fe1"]
        print("FEATURE_SOURCE:", args.feature_npz)
    else:
        fe7 = load_input(args.model_path, DEC6, 0).astype(np.float32)
        fe6 = make_feature((1, 96, 8, 8), 6).astype(np.float32)
        fe5 = make_feature((1, 48, 16, 16), 5).astype(np.float32)
        fe4 = make_feature((1, 24, 32, 32), 4).astype(np.float32)
        fe3 = make_feature((1, 128, 64, 64), 3).astype(np.float32)
        fe2 = make_feature((1, 64, 128, 128), 2).astype(np.float32)
        fe1 = make_feature((1, 64, 128, 128), 1).astype(np.float32)
        print("FEATURE_SOURCE: synthetic")

    global ENABLE_DEC6_BARRIER, BARRIER_DEC6_FLOAT, USE_SECOND_RUN, SECOND_RUN_MODE
    ENABLE_DEC6_BARRIER = bool(args.dec6_barrier_before_each)
    BARRIER_DEC6_FLOAT = fe7
    USE_SECOND_RUN = bool(args.use_second_run)
    SECOND_RUN_MODE = args.second_run_mode
    global CLEAR_WR_DONE_BEFORE_RUN
    CLEAR_WR_DONE_BEFORE_RUN = bool(args.clear_wr_done_before_run)
    if ENABLE_DEC6_BARRIER:
        print("DEC6_BARRIER_BEFORE_EACH: True")
    if USE_SECOND_RUN:
        print("USE_SECOND_RUN: True")
    if SECOND_RUN_MODE != "none":
        print("SECOND_RUN_MODE:", SECOND_RUN_MODE)
    if CLEAR_WR_DONE_BEFORE_RUN:
        init_pl_mmap()
        print("CLEAR_WR_DONE_BEFORE_RUN: True")

    try:
        fd6_raw, fd6_f = timed("dec6_rhb", lambda: run_scaled(DEC6, fe7), stats)
        fd5_raw, fd5_f = timed("dec5_rhb", lambda: run_scaled(DEC5, concat_like(fd6_f, fe6)), stats)
        fd4_raw, fd4_f = timed("dec4_rhb", lambda: run_scaled(DEC4, concat_like(fd5_f, fe5)), stats)
        fd3_raw, fd3_f = timed("dec3_rhb", lambda: run_scaled(DEC3, concat_like(fd4_f, fe4)), stats)
        if args.pre_dec2_dummy == "dec6":
            _dummy_raw, _dummy_f = timed("pre_dec2_dummy_dec6_rhb", lambda: run_scaled(DEC6, fe7), stats)
        elif args.pre_dec2_dummy == "dec5":
            _dummy_raw, _dummy_f = timed("pre_dec2_dummy_dec5_rhb", lambda: run_scaled(DEC5, concat_like(fd6_f, fe6)), stats)
        elif args.pre_dec2_dummy == "dec4":
            _dummy_raw, _dummy_f = timed("pre_dec2_dummy_dec4_rhb", lambda: run_scaled(DEC4, concat_like(fd5_f, fe5)), stats)
        elif args.pre_dec2_dummy == "dec3":
            _dummy_raw, _dummy_f = timed("pre_dec2_dummy_dec3_rhb", lambda: run_scaled(DEC3, concat_like(fd4_f, fe4)), stats)
        if args.reset_before_dec2:
            print("RESET_BEFORE_DEC2: True")
            ac_driver.load_model(args.model_path)
            load_activation_scales(args.model_path)
        dec2_trace = {}
        fd2_f = dec2_resize_upconv_split(
            fd3_f,
            fe3,
            stats,
            dec2_trace,
            args.warmup_dec2_up0,
            args.warmup_dec2_all,
            args.dec2_mode,
        )

        head_in = concat_like(fd2_f, fe2)
        if args.warmup_head_all:
            _warm_raw, _warm_f = timed("warmup_dep_dec1_full_rhb", lambda: run_scaled(DEP_DEC1_FULL, head_in), stats)
        dep_fd1_raw, dep_fd1_f = timed("dep_dec1_full_rhb", lambda: run_scaled(DEP_DEC1_FULL, head_in), stats)
        dep_dec0_in = concat_like(dep_fd1_f, fe1)
        if args.warmup_head_dec0 or args.warmup_head_all:
            _warm_raw, _warm_f = timed("warmup_dep_dec0_rhb", lambda: run_scaled(DEP_DEC0, dep_dec0_in), stats)
        init_depth_raw, init_depth_f = timed("dep_dec0_rhb", lambda: run_scaled(DEP_DEC0, dep_dec0_in), stats)
        if args.warmup_head_all:
            _warm_raw, _warm_f = timed("warmup_gd_dec1_full_rhb", lambda: run_scaled(GD_DEC1_FULL, head_in), stats)
        gd_fd1_raw, gd_fd1_f = timed("gd_dec1_full_rhb", lambda: run_scaled(GD_DEC1_FULL, head_in), stats)
        gd_dec0_in = concat_like(gd_fd1_f, fe1)
        if args.warmup_head_dec0 or args.warmup_head_all:
            _warm_raw, _warm_f = timed("warmup_gd_dec0_rhb", lambda: run_scaled(GD_DEC0, gd_dec0_in), stats)
        guide_raw, guide_f = timed("gd_dec0_rhb", lambda: run_scaled(GD_DEC0, gd_dec0_in), stats)
        if args.warmup_head_all:
            _warm_raw, _warm_f = timed("warmup_cf_dec1_rhb", lambda: run_scaled(CF_DEC1, head_in), stats)
        cf_fd1_raw, cf_fd1_f = timed("cf_dec1_rhb", lambda: run_scaled(CF_DEC1, head_in), stats)
        cf_dec0_in = concat_like(cf_fd1_f, fe1)
        cf_dec0_model = CF_DEC0_CONV_ONLY if args.cf_dec0_host_sigmoid else CF_DEC0
        if args.warmup_head_dec0 or args.warmup_head_all:
            _warm_raw, _warm_f = timed("warmup_cf_dec0_rhb", lambda: run_scaled(cf_dec0_model, cf_dec0_in), stats)
        confidence_raw, confidence_logits_f = timed("cf_dec0_rhb", lambda: run_scaled(cf_dec0_model, cf_dec0_in), stats)
        if args.cf_dec0_host_sigmoid:
            confidence_f = timed("cf_dec0_host_sigmoid", lambda: sigmoid_float(confidence_logits_f), stats)
        else:
            confidence_f = confidence_logits_f
    finally:
        close_pl_mmap()

    fd6 = quant_input(fd6_f, OUTPUT_SCALES[DEC6])
    fd5 = quant_input(fd5_f, OUTPUT_SCALES[DEC5])
    fd4 = quant_input(fd4_f, OUTPUT_SCALES[DEC4])
    fd3 = quant_input(fd3_f, OUTPUT_SCALES[DEC3])
    fd2 = quant_input(fd2_f, INPUT_SCALES[DEP_DEC1_FULL])
    dep_fd1 = quant_input(dep_fd1_f, OUTPUT_SCALES[DEP_DEC1_FULL])
    init_depth = quant_input(init_depth_f, OUTPUT_SCALES[DEP_DEC0])
    gd_fd1 = quant_input(gd_fd1_f, OUTPUT_SCALES[GD_DEC1_FULL])
    guide = quant_input(guide_f, OUTPUT_SCALES[GD_DEC0])
    cf_fd1 = quant_input(cf_fd1_f, OUTPUT_SCALES[CF_DEC1])
    confidence_scale = 128.0 if args.cf_dec0_host_sigmoid else OUTPUT_SCALES[CF_DEC0]
    confidence = quant_input(confidence_f, confidence_scale)

    if args.save_npz:
        np.savez(
            args.save_npz,
            feature_npz=np.array([args.feature_npz]),
            fd6=fd6, fd5=fd5, fd4=fd4, fd3=fd3, fd2=fd2,
            dep_fd1=dep_fd1, init_depth=init_depth,
            gd_fd1=gd_fd1, guide=guide, cf_fd1=cf_fd1, confidence=confidence,
            fd6_raw=fd6_raw, fd5_raw=fd5_raw, fd4_raw=fd4_raw, fd3_raw=fd3_raw,
            dep_fd1_raw=dep_fd1_raw, init_depth_raw=init_depth_raw,
            gd_fd1_raw=gd_fd1_raw, guide_raw=guide_raw, cf_fd1_raw=cf_fd1_raw, confidence_raw=confidence_raw,
            fd6_f=fd6_f, fd5_f=fd5_f, fd4_f=fd4_f, fd3_f=fd3_f, fd2_f=fd2_f,
            dep_fd1_f=dep_fd1_f, init_depth_f=init_depth_f,
            gd_fd1_f=gd_fd1_f, guide_f=guide_f, cf_fd1_f=cf_fd1_f, confidence_f=confidence_f,
            confidence_logits_f=confidence_logits_f,
            **dec2_trace,
        )
        print("SAVED_NPZ:", args.save_npz)

    print("Decoder fd6/fd5/fd4/fd3/fd2 shapes:", fd6.shape, fd5.shape, fd4.shape, fd3.shape, fd2.shape)
    print("Decoder head output shapes:", init_depth.shape, guide.shape, confidence.shape)
    print("Decoder output min/max:", int(init_depth.min()), int(init_depth.max()), int(guide.min()), int(guide.max()), int(confidence.min()), int(confidence.max()))
    print(f"LATENCY decoder_system_128x128_ckpt_tracked_total: {sum(v for _, v in stats):.3f} ms")
    print("DECODER_SYSTEM_128X128_CKPT_EXECUTED: True")
    print("CKPT_WEIGHTS: ckpt-backed exported submodels; dec2 resize+upconv on RHB, dep/gd dec1 full conv")


if __name__ == "__main__":
    main()
