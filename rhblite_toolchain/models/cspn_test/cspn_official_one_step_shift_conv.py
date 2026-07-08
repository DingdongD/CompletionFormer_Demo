import torch
import torch.nn as nn

input_layouts = ["CHW", "CHW", "CHW", "CHW"]
ifmap_sz = [[1, 32, 32], [8, 32, 32], [1, 32, 32], [1, 32, 32]]
op_version = 14
batch_size = 1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_sum = nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False)

        weight = torch.zeros(1, 8, 3, 3)
        # Official CSPN direction order:
        # left_top, center_top, right_top, left_center, right_center,
        # left_bottom, center_bottom, right_bottom.
        #
        # These coordinates are equivalent to:
        # pad each direction with the official ZeroPad2d config,
        # multiply gate/depth on the padded canvas,
        # sum over 8 directions, then crop [1:-1, 1:-1].
        coords = [
            (2, 2), (2, 1), (2, 0),
            (1, 2),         (1, 0),
            (0, 2), (0, 1), (0, 0),
        ]
        for k, (r, c) in enumerate(coords):
            weight[0, k, r, c] = 1.0
        with torch.no_grad():
            self.shift_sum.weight.copy_(weight)
        self.shift_sum.weight.requires_grad_(False)

    def forward(self, depth, gate8, raw_depth, center):
        weighted_depth = depth * gate8
        neighbor_weighted_sum = self.shift_sum(weighted_depth)
        center_residual = raw_depth * center
        return center_residual + neighbor_weighted_sum
