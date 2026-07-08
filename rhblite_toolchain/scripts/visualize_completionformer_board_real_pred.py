import argparse
import os
import os.path as osp
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "CompletionFormer"))


def denorm_rgb(rgb):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    x = rgb * std + mean
    return np.clip(np.transpose(x, (1, 2, 0)), 0.0, 1.0)


def load_args_and_prop_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = deepcopy(ckpt["args"])
    args.augment = False
    prop_state = {
        key[len("prop_layer.") :]: value
        for key, value in ckpt["net"].items()
        if key.startswith("prop_layer.")
    }
    return args, prop_state


def run_host_nlspn_from_board(board_npz, feature_npz, ckpt_path):
    from ref_model_hw.nlspn_hw import NLSPNHW

    args, prop_state = load_args_and_prop_state(ckpt_path)
    feat = np.load(feature_npz)
    board = np.load(board_npz)

    dep = torch.from_numpy(feat["dep"].astype(np.float32))
    rgb = torch.from_numpy(feat["rgb"].astype(np.float32))
    init_key = "init_depth_f" if "init_depth_f" in board.files else "init_depth"
    guide_key = "guide_f" if "guide_f" in board.files else "guide"
    conf_key = "confidence_f" if "confidence_f" in board.files else "confidence"
    init_depth_raw = torch.from_numpy(board[init_key].astype(np.float32))
    guide = torch.from_numpy(board[guide_key].astype(np.float32))
    confidence = torch.from_numpy(board[conf_key].astype(np.float32))
    pred_init = init_depth_raw + dep

    model = NLSPNHW(args, args.prop_kernel * args.prop_kernel - 1, 1, 3, args.prop_kernel).eval()
    model.load_state_dict(prop_state, strict=True)
    with torch.no_grad():
        pred, pred_inter, offset, aff, gamma = model(pred_init, guide, confidence, dep, rgb)
        pred = torch.clamp(pred, min=0)

    return {
        "pred": pred.detach().cpu().numpy().astype(np.float32),
        "pred_init": pred_init.detach().cpu().numpy().astype(np.float32),
        "offset": offset.detach().cpu().numpy().astype(np.float32),
        "aff": aff.detach().cpu().numpy().astype(np.float32),
        "gamma": gamma.detach().cpu().numpy().astype(np.float32),
        "pred_inter_count": np.array([len(pred_inter)], dtype=np.int32),
    }


def image_depth(ax, arr, title, vmin=None, vmax=None):
    im = ax.imshow(arr, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-npz", required=True)
    parser.add_argument("--board-npz", required=True)
    parser.add_argument("--ckpt", default=osp.join(ROOT, "CompletionFormer/ref_model_hw/model_00030.pt"))
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-npz", default="")
    args = parser.parse_args()

    feat = np.load(args.feature_npz)
    board = np.load(args.board_npz)
    host = run_host_nlspn_from_board(args.board_npz, args.feature_npz, args.ckpt)

    rgb = feat["rgb"][0]
    dep = feat["dep"][0, 0]
    gt = feat["gt"][0, 0]
    ref_pred = feat["pred_ref"][0, 0]
    ref_pred_init = feat["pred_init_ref"][0, 0]
    board_init_key = "init_depth_f" if "init_depth_f" in board.files else "init_depth"
    board_init_raw = board[board_init_key].astype(np.float32)[0, 0]
    board_pred_init = host["pred_init"][0, 0]
    board_pred = host["pred"][0, 0]

    vmin = float(min(gt.min(), ref_pred.min()))
    vmax = float(max(gt.max(), ref_pred.max()))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    axes[0, 0].imshow(denorm_rgb(rgb))
    axes[0, 0].set_title("rgb", fontsize=9)
    axes[0, 0].axis("off")
    image_depth(axes[0, 1], dep, "sparse dep", vmin=vmin, vmax=vmax)
    image_depth(axes[0, 2], gt, "gt", vmin=vmin, vmax=vmax)
    image_depth(axes[0, 3], ref_pred, "ref pred", vmin=vmin, vmax=vmax)
    image_depth(axes[1, 0], ref_pred_init, "ref pred_init", vmin=vmin, vmax=vmax)
    image_depth(axes[1, 1], board_init_raw, "board init float")
    image_depth(axes[1, 2], board_pred_init, "board pred_init")
    image_depth(axes[1, 3], board_pred, "board pred")

    os.makedirs(osp.dirname(osp.abspath(args.out_png)), exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)

    if args.out_npz:
        np.savez(
            args.out_npz,
            rgb=feat["rgb"],
            dep=feat["dep"],
            gt=feat["gt"],
            ref_pred=feat["pred_ref"],
            ref_pred_init=feat["pred_init_ref"],
            board_init_depth=board[board_init_key],
            board_guide=board["guide_f"] if "guide_f" in board.files else board["guide"],
            board_confidence=board["confidence_f"] if "confidence_f" in board.files else board["confidence"],
            board_pred=host["pred"],
            board_pred_init=host["pred_init"],
            board_offset=host["offset"],
            board_aff=host["aff"],
            board_gamma=host["gamma"],
        )

    print("BOARD_REAL_PRED_VIS:", args.out_png)
    if args.out_npz:
        print("BOARD_REAL_PRED_NPZ:", args.out_npz)
    print("ref_pred min/max:", float(ref_pred.min()), float(ref_pred.max()))
    print("board_init_raw min/max:", float(board_init_raw.min()), float(board_init_raw.max()))
    print("board_pred_raw min/max:", float(board_pred.min()), float(board_pred.max()))
    print("board_npz:", args.board_npz)
    print("feature_npz:", args.feature_npz)


if __name__ == "__main__":
    main()
