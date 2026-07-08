#!/usr/bin/env python3
import torch
import torch.nn.functional as F


def original_eight_neighbors_replicate(x):
    _, _, h, w = x.shape
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return torch.cat([
        p[:, :, 0:h,     0:w],
        p[:, :, 0:h,     1:w + 1],
        p[:, :, 0:h,     2:w + 2],
        p[:, :, 1:h + 1, 0:w],
        p[:, :, 1:h + 1, 2:w + 2],
        p[:, :, 2:h + 2, 0:w],
        p[:, :, 2:h + 2, 1:w + 1],
        p[:, :, 2:h + 2, 2:w + 2],
    ], dim=1)


def original_eight_neighbors_zero(x):
    _, _, h, w = x.shape
    p = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)
    return torch.cat([
        p[:, :, 0:h,     0:w],
        p[:, :, 0:h,     1:w + 1],
        p[:, :, 0:h,     2:w + 2],
        p[:, :, 1:h + 1, 0:w],
        p[:, :, 1:h + 1, 2:w + 2],
        p[:, :, 2:h + 2, 0:w],
        p[:, :, 2:h + 2, 1:w + 1],
        p[:, :, 2:h + 2, 2:w + 2],
    ], dim=1)


def fixed_neighbor_conv(x, padding_mode="zeros"):
    conv = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode)
    weight = torch.zeros(8, 1, 3, 3)
    coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    for i, (r, c) in enumerate(coords):
        weight[i, 0, r, c] = 1.0
    with torch.no_grad():
        conv.weight.copy_(weight)
    return conv(x)


def one_step_from_neighbors(depth0, affinity8, center, neighbors):
    return depth0 * center + (neighbors * affinity8).sum(dim=1, keepdim=True)


def one_step_no_concat_conv(depth0, affinity8, center, padding_mode="zeros"):
    nbs = fixed_neighbor_conv(depth0, padding_mode=padding_mode)
    sum_conv = torch.nn.Conv2d(8, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        sum_conv.weight.fill_(1.0)
    return depth0 * center + sum_conv(nbs * affinity8)


def report(name, a, b):
    diff = (a - b).abs()
    print(f"{name:42s} max_abs={diff.max().item():.8g} mean_abs={diff.mean().item():.8g}")


def main():
    torch.manual_seed(20260622)
    depth = torch.randn(2, 1, 5, 6)
    affinity8 = torch.randn(2, 8, 5, 6)
    center = torch.randn(2, 1, 5, 6)

    orig_rep = original_eight_neighbors_replicate(depth)
    orig_zero = original_eight_neighbors_zero(depth)
    conv_zero = fixed_neighbor_conv(depth, padding_mode="zeros")
    conv_rep = fixed_neighbor_conv(depth, padding_mode="replicate")

    print("Eight-neighbor mapping:")
    report("original zero-pad vs fixed conv zero", orig_zero, conv_zero)
    report("original replicate vs fixed conv replicate", orig_rep, conv_rep)
    report("original replicate vs fixed conv zero", orig_rep, conv_zero)

    interior = (..., slice(1, -1), slice(1, -1))
    print("\nInterior only:")
    report("replicate vs zero on interior", orig_rep[interior], conv_zero[interior])

    print("\nOne-step propagation:")
    ref_rep = one_step_from_neighbors(depth, affinity8, center, orig_rep)
    ref_zero = one_step_from_neighbors(depth, affinity8, center, orig_zero)
    conv_step_zero = one_step_no_concat_conv(depth, affinity8, center, padding_mode="zeros")
    conv_step_rep = one_step_no_concat_conv(depth, affinity8, center, padding_mode="replicate")
    report("original zero one-step vs no-concat conv", ref_zero, conv_step_zero)
    report("original replicate one-step vs conv replicate", ref_rep, conv_step_rep)
    report("original replicate one-step vs conv zero", ref_rep, conv_step_zero)
    report("replicate vs zero one-step interior", ref_rep[..., 1:-1, 1:-1], conv_step_zero[..., 1:-1, 1:-1])


if __name__ == "__main__":
    main()
