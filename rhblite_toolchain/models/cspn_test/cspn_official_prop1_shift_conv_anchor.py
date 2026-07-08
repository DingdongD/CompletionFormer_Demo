import torch
import torch.nn as nn

input_layouts = ["CHW", "CHW", "CHW", "CHW", "CHW"]
ifmap_sz = [[1, 32, 32], [8, 32, 32], [1, 32, 32], [1, 32, 32], [1, 32, 32]]
op_version = 14
batch_size = 1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_sum = nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False)
        weight = torch.zeros(1, 8, 3, 3)
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

    def forward(self, raw_depth, gate8, center, mask, inv_mask):
        propagated = raw_depth * center + self.shift_sum(raw_depth * gate8)
        return inv_mask * propagated + mask * raw_depth
