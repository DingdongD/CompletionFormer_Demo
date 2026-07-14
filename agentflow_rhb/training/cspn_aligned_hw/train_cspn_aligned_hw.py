import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.append(".")
sys.path.insert(0, "./models")

import nyu_dataset_loader as dataset_loader
from cspn_aligned_hw import AlignedCSPNHW
from cspn_resnet_tiny_aligned_hw import ResNetTinyAlignedCSPNHW


def resize_batch(rgbd, depth, image_size):
    rgb = F.interpolate(rgbd[:, :3], size=(image_size, image_size), mode="bilinear", align_corners=False)
    sparse = F.interpolate(rgbd[:, 3:4], size=(image_size, image_size), mode="nearest")
    target = F.interpolate(depth, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return torch.cat([rgb, sparse], dim=1), target


def masked_l1(pred, target):
    valid = (target > 1.0e-4).float()
    denom = valid.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * valid).sum() / denom


def rmse(pred, target):
    valid = target > 1.0e-4
    if not bool(valid.any()):
        return 0.0
    return torch.sqrt(torch.mean((pred[valid] - target[valid]) ** 2)).item()


def load_model_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model_state = model.state_dict()
    has_module = next(iter(model_state)).startswith("module.") if model_state else False
    ckpt_has_module = next(iter(state)).startswith("module.") if state else False
    if has_module and not ckpt_has_module:
        state = {f"module.{key}": value for key, value in state.items()}
    elif ckpt_has_module and not has_module:
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def load_qat_scales(path, column):
    if not path:
        return None
    scales = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            scales[row["name"]] = float(row[column])
    return scales


def build_loaders(args):
    trainset = dataset_loader.NyuDepthDataset(
        csv_file=args.train_list,
        root_dir=".",
        split="train",
        n_sample=args.n_sample,
        input_format="hdf5",
    )
    valset = dataset_loader.NyuDepthDataset(
        csv_file=args.eval_list,
        root_dir=".",
        split="val",
        n_sample=args.n_sample,
        input_format="hdf5",
    )
    if 0 < args.subset < len(trainset):
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(trainset), generator=generator)[: args.subset].tolist()
        trainset = Subset(trainset, indices)
    if 0 < args.val_subset < len(valset):
        valset = Subset(valset, list(range(args.val_subset)))
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(model, loader, args, device):
    model.eval()
    total_l1 = 0.0
    total_rmse = 0.0
    count = 0
    with torch.no_grad():
        for sample in loader:
            rgbd, target = resize_batch(sample["rgbd"].to(device), sample["depth"].to(device), args.image_size)
            pred = model(rgbd)
            total_l1 += masked_l1(pred, target).item()
            total_rmse += rmse(pred, target)
            count += 1
    count = max(count, 1)
    return {"val_l1": total_l1 / count, "val_rmse": total_rmse / count}


def main():
    parser = argparse.ArgumentParser(description="Train RHB-aligned CSPN HW model on NYU")
    parser.add_argument("--arch", choices=["toy", "resnet_tiny"], default="toy")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--scheduler-step", type=int, default=0)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--base-ch", type=int, default=8)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--cspn-step", type=int, default=4)
    parser.add_argument("--cspn-norm-type", default="8sum")
    parser.add_argument("--down20-variant", choices=["stride3x3", "sample_1x1"], default="stride3x3")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--raw-depth-loss-weight", type=float, default=0.2)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--subset", type=int, default=4096)
    parser.add_argument("--val-subset", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-list", default="datalist/nyudepth_hdf5_train.csv")
    parser.add_argument("--eval-list", default="datalist/nyudepth_hdf5_val.csv")
    parser.add_argument("--save-dir", default="output/nyu_cspn_aligned_hw32")
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-weights-only", action="store_true")
    parser.add_argument("--fused-contract-qat", action="store_true")
    parser.add_argument("--qat-scales-csv", default="")
    parser.add_argument("--qat-scale-column", default="recommended_scale_max")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qat_scales = load_qat_scales(args.qat_scales_csv, args.qat_scale_column)
    if args.arch == "toy":
        model = AlignedCSPNHW(
            base_ch=args.base_ch,
            cspn_step=args.cspn_step,
            cspn_norm_type=args.cspn_norm_type,
            down20_variant=args.down20_variant,
        ).to(device)
    else:
        model = ResNetTinyAlignedCSPNHW(
            width=args.width,
            cspn_step=args.cspn_step,
            cspn_norm_type=args.cspn_norm_type,
            downsample_variant=args.down20_variant,
            fused_contract_qat=args.fused_contract_qat,
            qat_scales=qat_scales,
        ).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    resume_checkpoint = None
    start_epoch = 0
    if args.resume:
        resume_checkpoint = load_model_state(model, args.resume, device)
        if args.resume_weights_only:
            start_epoch = 0
        else:
            start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1 if isinstance(resume_checkpoint, dict) else 0
        print(f"loaded resume checkpoint={args.resume} start_epoch={start_epoch} weights_only={args.resume_weights_only}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    if (not args.resume_weights_only) and isinstance(resume_checkpoint, dict) and "optimizer_state" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        print("loaded optimizer state", flush=True)
    scheduler = None
    if args.scheduler_step > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step,
            gamma=args.scheduler_gamma,
        )
        if (not args.resume_weights_only) and isinstance(resume_checkpoint, dict) and "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            print("loaded scheduler state", flush=True)
    train_loader, val_loader = build_loaders(args)

    metadata = {
        "model": "AlignedCSPNHW" if args.arch == "toy" else "ResNetTinyAlignedCSPNHW",
        "arch": args.arch,
        "base_ch": args.base_ch,
        "width": args.width,
        "cspn_step": args.cspn_step,
        "down20_variant": args.down20_variant,
        "image_size": args.image_size,
        "raw_depth_loss_weight": args.raw_depth_loss_weight,
        "data_parallel": bool(args.data_parallel and torch.cuda.device_count() > 1),
        "cuda_device_count": torch.cuda.device_count(),
        "n_sample": args.n_sample,
        "train_subset": args.subset,
        "val_subset": args.val_subset,
        "resume": args.resume,
        "resume_weights_only": args.resume_weights_only,
        "fused_contract_qat": args.fused_contract_qat,
        "qat_scales_csv": args.qat_scales_csv,
        "qat_scale_column": args.qat_scale_column,
        "start_epoch": start_epoch,
        "lr": args.lr,
        "scheduler_step": args.scheduler_step,
        "scheduler_gamma": args.scheduler_gamma,
    }
    Path(args.save_dir, "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best = float("inf")
    if isinstance(resume_checkpoint, dict) and "metrics" in resume_checkpoint:
        best = float(resume_checkpoint["metrics"].get("val_rmse", best))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        start = time.time()
        loss_sum = 0.0
        for step, sample in enumerate(train_loader):
            rgbd, target = resize_batch(sample["rgbd"].to(device), sample["depth"].to(device), args.image_size)
            optimizer.zero_grad(set_to_none=True)
            pred = model(rgbd)
            loss = masked_l1(pred, target)
            if args.raw_depth_loss_weight > 0:
                raw_model = model.module if isinstance(model, nn.DataParallel) else model
                raw_depth, _ = raw_model.forward_heads(rgbd)
                loss = loss + args.raw_depth_loss_weight * masked_l1(raw_depth, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if step % 20 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss_sum / (step + 1):.6f}", flush=True)
        metrics = evaluate(model, val_loader, args, device)
        train_loss = loss_sum / max(len(train_loader), 1)
        elapsed = time.time() - start
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_l1={metrics['val_l1']:.6f} "
            f"val_rmse={metrics['val_rmse']:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "metadata": metadata,
            "metrics": metrics,
        }
        torch.save(state, Path(args.save_dir, f"epoch_{epoch:03d}.pt"))
        if metrics["val_rmse"] < best:
            best = metrics["val_rmse"]
            torch.save(state, Path(args.save_dir, "best_model.pt"))
            print(f"saved best_model.pt val_rmse={best:.6f}", flush=True)
        if scheduler is not None:
            scheduler.step()
    print(f"done best_val_rmse={best:.6f} save_dir={args.save_dir}")


if __name__ == "__main__":
    main()
