#!/usr/bin/env python3
import torch
import torch.nn.functional as F

PADS = (
    (0, 2, 0, 2),
    (1, 1, 0, 2),
    (2, 0, 0, 2),
    (0, 2, 1, 1),
    (2, 0, 1, 1),
    (0, 2, 2, 0),
    (1, 1, 2, 0),
    (2, 0, 2, 0),
)


def pad_one(x, k):
    return F.pad(x, PADS[k]).unsqueeze(1)


def official_no_norm_one_step(depth, gate8, raw_depth, center):
    depth_pad = torch.cat([pad_one(depth, k) for k in range(8)], dim=1)
    gate_pad = torch.cat([pad_one(gate8[:, k:k + 1], k) for k in range(8)], dim=1)
    neighbor = (depth_pad * gate_pad).sum(dim=1)[:, :, 1:-1, 1:-1]
    return center * raw_depth + neighbor


def conv_shift_sum_one_step(depth, gate8, raw_depth, center):
    prod = depth * gate8
    conv = torch.nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False)
    weight = torch.zeros(1, 8, 3, 3)
    # Equivalent to official ZeroPad2d direction pads followed by crop [1:-1, 1:-1].
    coords = [
        (2, 2), (2, 1), (2, 0),
        (1, 2),         (1, 0),
        (0, 2), (0, 1), (0, 0),
    ]
    for k, (r, c) in enumerate(coords):
        weight[0, k, r, c] = 1.0
    with torch.no_grad():
        conv.weight.copy_(weight)
    return center * raw_depth + conv(prod)


def report(name, a, b):
    diff = (a - b).abs()
    print(f"{name:36s} max_abs={diff.max().item():.8g} mean_abs={diff.mean().item():.8g}")


def main():
    torch.manual_seed(20260622)
    depth = torch.randn(2, 1, 5, 6)
    raw_depth = torch.randn(2, 1, 5, 6)
    gate8 = torch.randn(2, 8, 5, 6)
    center = torch.randn(2, 1, 5, 6)
    official = official_no_norm_one_step(depth, gate8, raw_depth, center)
    conv = conv_shift_sum_one_step(depth, gate8, raw_depth, center)
    report("official pad/crop vs fixed conv", official, conv)


if __name__ == "__main__":
    main()
