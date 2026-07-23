#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def import_remote_modules(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "models"))
    sys.path.insert(0, str(repo_root / "nlspn_test"))


def resize_batch(rgbd: torch.Tensor, depth: torch.Tensor, image_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = F.interpolate(rgbd[:, :3], size=(image_size, image_size), mode="bilinear", align_corners=False)
    sparse = F.interpolate(rgbd[:, 3:4], size=(image_size, image_size), mode="nearest")
    target = F.interpolate(depth, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return rgb, sparse, target


def masked_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = (target > 1.0e-4).float()
    return (torch.abs(pred - target) * valid).sum() / valid.sum().clamp_min(1.0)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    valid = target > 1.0e-4
    if not bool(valid.any()):
        return 0.0
    return torch.sqrt(torch.mean((pred[valid] - target[valid]) ** 2)).item()


def build_loaders(args: argparse.Namespace):
    import nyu_dataset_loader as dataset_loader

    trainset = dataset_loader.NyuDepthDataset(
        csv_file=args.train_list,
        root_dir=args.data_root,
        split="train",
        n_sample=args.n_sample,
        input_format="hdf5",
    )
    valset = dataset_loader.NyuDepthDataset(
        csv_file=args.eval_list,
        root_dir=args.data_root,
        split="val",
        n_sample=args.n_sample,
        input_format="hdf5",
    )
    if 0 < args.subset < len(trainset):
        generator = torch.Generator().manual_seed(args.seed)
        trainset = Subset(trainset, torch.randperm(len(trainset), generator=generator)[: args.subset].tolist())
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


def evaluate(model: nn.Module, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict:
    model.eval()
    total_l1 = 0.0
    total_init_l1 = 0.0
    total_rmse = 0.0
    count = 0
    with torch.no_grad():
        for sample in loader:
            rgb, sparse, target = resize_batch(sample["rgbd"].to(device), sample["depth"].to(device), args.image_size)
            out = model({"rgb": rgb, "dep": sparse})
            total_l1 += masked_l1(out["pred"], target).item()
            total_init_l1 += masked_l1(out["pred_init"], target).item()
            total_rmse += rmse(out["pred"], target)
            count += 1
    count = max(count, 1)
    return {
        "val_l1": total_l1 / count,
        "val_pred_init_l1": total_init_l1 / count,
        "val_rmse": total_rmse / count,
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int, metrics: dict, args: argparse.Namespace) -> None:
    module = model.module if isinstance(model, nn.DataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NLSPN HW-aligned model with CSPN NYU h5 data")
    parser.add_argument("--repo-root", default="/workspace/CSPN/cspn_pytorch")
    parser.add_argument("--data-root", default="/workspace/CSPN/cspn_pytorch")
    parser.add_argument("--train-list", default="/workspace/CSPN/cspn_pytorch/datalist/nyudepth_hdf5_train.csv")
    parser.add_argument("--eval-list", default="/workspace/CSPN/cspn_pytorch/datalist/nyudepth_hdf5_val.csv")
    parser.add_argument("--save-dir", default="/workspace/CSPN/cspn_pytorch/output/nyu_nlspn_hw_aligned_128")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--scheduler-step", type=int, default=20)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--val-subset", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--prop-time", type=int, default=6)
    parser.add_argument("--preserve-input", action="store_true")
    parser.add_argument("--pred-init-loss-weight", type=float, default=0.35)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    import_remote_modules(Path(args.repo_root))
    from nlspn_test.nlspn_hw_aligned import NLSPNHWAlignedModel, NLSPNHWConfig

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metadata.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NLSPNHWAlignedModel(
        NLSPNHWConfig(
            input_height=args.image_size,
            input_width=args.image_size,
            network="resnet18",
            from_scratch=True,
            prop_time=args.prop_time,
            preserve_input=args.preserve_input,
            allow_approx_fixed_neighbor=True,
        )
    ).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        if args.scheduler_step > 0
        else None
    )
    start_epoch = 0
    best_l1 = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.load_state_dict(state, strict=True)
        if isinstance(ckpt, dict) and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and isinstance(ckpt, dict) and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 0
        best_l1 = float(ckpt.get("metrics", {}).get("best_l1", best_l1)) if isinstance(ckpt, dict) else best_l1

    train_loader, val_loader = build_loaders(args)
    log_path = save_dir / "log_train.txt"
    print(f"device={device} gpus={torch.cuda.device_count()} train_batches={len(train_loader)} val_batches={len(val_loader)}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, sample in enumerate(train_loader):
            rgb, sparse, target = resize_batch(sample["rgbd"].to(device), sample["depth"].to(device), args.image_size)
            out = model({"rgb": rgb, "dep": sparse})
            loss_pred = masked_l1(out["pred"], target)
            loss_init = masked_l1(out["pred_init"], target)
            loss = loss_pred + args.pred_init_loss_weight * loss_init
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            if step % 50 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.5f} "
                    f"pred={loss_pred.item():.5f} init={loss_init.item():.5f}",
                    flush=True,
                )
        if scheduler is not None:
            scheduler.step()
        metrics = evaluate(model, val_loader, args, device)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": running / max(len(train_loader), 1),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_sec": time.time() - t0,
                "best_l1": best_l1,
            }
        )
        if metrics["val_l1"] < best_l1:
            best_l1 = metrics["val_l1"]
            metrics["best_l1"] = best_l1
            save_checkpoint(save_dir / "model_best.pt", model, optimizer, scheduler, epoch, metrics, args)
        save_checkpoint(save_dir / "model_latest.pt", model, optimizer, scheduler, epoch, metrics, args)
        line = json.dumps(metrics, sort_keys=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


if __name__ == "__main__":
    main()
