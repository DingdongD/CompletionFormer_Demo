import torch
import torch.nn as nn
import torch.nn.functional as F


def modulated_deform_conv_fallback(input, offset, mask, weight, bias, padding):
    """Differentiable CPU/GPU fallback for NLSPN inference/training.

    This implements the `ch_f == 1`, stride=1, dilation=1, groups=1 case used
    by CompletionFormer NLSPN. It replaces the external DCN extension in the
    HW-aligned reference so NLSPN remains part of the model graph.
    """

    if input.shape[1] != 1 or weight.shape[:2] != (1, 1):
        raise NotImplementedError("NLSPNHW fallback currently supports ch_f == 1 only")

    bsz, _, height, width = input.shape
    kernel_h, kernel_w = weight.shape[-2:]
    dtype = input.dtype
    device = input.device

    ys = torch.arange(height, device=device, dtype=dtype).view(1, height, 1).expand(bsz, height, width)
    xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, width).expand(bsz, height, width)

    out = input.new_zeros((bsz, 1, height, width))
    idx = 0
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            off_y = offset[:, 2 * idx + 0]
            off_x = offset[:, 2 * idx + 1]
            sample_y = ys + kh - padding + off_y
            sample_x = xs + kw - padding + off_x
            norm_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
            norm_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
            grid = torch.stack((norm_x, norm_y), dim=-1)
            sampled = F.grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            out = out + sampled * mask[:, idx : idx + 1] * weight[0, 0, kh, kw]
            idx += 1

    if bias is not None:
        out = out + bias.view(1, 1, 1, 1)
    return out


class NLSPNHW(nn.Module):
    """Host-side NLSPN kept inside the HW-aligned CompletionFormer graph."""

    def __init__(self, args, ch_g, ch_f, k_g, k_f):
        super().__init__()
        if ch_f != 1:
            raise ValueError(f"NLSPNHW only supports ch_f == 1, got {ch_f}")
        if k_g % 2 != 1 or k_f % 2 != 1:
            raise ValueError("NLSPNHW requires odd guidance/feature kernels")

        self.args = args
        self.prop_time = args.prop_time
        self.affinity = args.affinity
        self.ch_g = ch_g
        self.ch_f = ch_f
        self.k_g = k_g
        self.k_f = k_f
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2
        self.padding = (k_f - 1) // 2

        if self.affinity not in ["AS", "ASS", "TC", "TGASS"]:
            raise NotImplementedError(self.affinity)

        self.conv_offset_aff = nn.Conv2d(
            self.ch_g,
            3 * self.num,
            kernel_size=self.k_g,
            stride=1,
            padding=(self.k_g - 1) // 2,
            bias=True,
        )
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()

        if self.affinity == "TGASS":
            self.aff_scale_const = nn.Parameter(args.affinity_gamma * self.num * torch.ones(1))
        elif self.affinity == "TC":
            self.aff_scale_const = nn.Parameter(self.num * torch.ones(1), requires_grad=False)
        else:
            self.aff_scale_const = nn.Parameter(torch.ones(1), requires_grad=False)

        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(self.ch_f), requires_grad=False)
        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)), requires_grad=False)

    def _deform(self, feat, offset, mask, weight):
        return modulated_deform_conv_fallback(feat, offset, mask, weight, self.b, self.padding)

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        bsz, _, height, width = guidance.shape
        offset_aff = self.conv_offset_aff(guidance)
        o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

        offset = torch.cat((o1, o2), dim=1).view(bsz, self.num, 2, height, width)
        list_offset = list(torch.chunk(offset, self.num, dim=1))
        list_offset.insert(self.idx_ref, torch.zeros((bsz, 1, 2, height, width), dtype=offset.dtype, device=offset.device))
        offset = torch.cat(list_offset, dim=1).view(bsz, -1, height, width)

        if self.affinity == "TC":
            aff = torch.tanh(aff / 100) / self.aff_scale_const
        elif self.affinity == "TGASS":
            aff = torch.tanh(aff / 100) / (self.aff_scale_const + 1e-8)

        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)
            modulation_dummy = torch.ones((bsz, 1, height, width), dtype=offset.dtype, device=offset.device)
            for idx_off in range(self.num + 1):
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f
                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].detach()
                if self.args.legacy:
                    offset_tmp = offset_tmp.clone()
                    offset_tmp[:, 0] = offset_tmp[:, 0] + hh - (self.k_f - 1) / 2
                    offset_tmp[:, 1] = offset_tmp[:, 1] + ww - (self.k_f - 1) / 2

                conf_tmp = modulated_deform_conv_fallback(
                    confidence,
                    offset_tmp,
                    modulation_dummy,
                    self.w_conf,
                    self.b,
                    padding=0,
                )
                list_conf.append(conf_tmp)
            aff = aff * torch.cat(list_conf, dim=1).contiguous()

        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4
        if self.affinity in ["ASS", "TGASS"]:
            aff_abs_sum = torch.where(aff_abs_sum < 1.0, torch.ones_like(aff_abs_sum), aff_abs_sum)
        if self.affinity in ["AS", "ASS", "TGASS"]:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum
        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        return self._deform(feat, offset, aff, self.w)

    def forward(self, feat_init, guidance, confidence=None, feat_fix=None, rgb=None):
        if self.ch_g != guidance.shape[1]:
            raise ValueError(f"Expected guidance channels {self.ch_g}, got {guidance.shape[1]}")
        if self.ch_f != feat_init.shape[1]:
            raise ValueError(f"Expected feat channels {self.ch_f}, got {feat_init.shape[1]}")
        if self.args.conf_prop and confidence is None:
            raise ValueError("confidence is required when conf_prop=True")

        offset, aff = self._get_offset_affinity(guidance, confidence if self.args.conf_prop else None, rgb)

        if self.args.preserve_input:
            if feat_fix is None or feat_init.shape != feat_fix.shape:
                raise ValueError("feat_fix with same shape is required when preserve_input=True")
            mask_fix = (torch.sum(feat_fix > 0.0, dim=1, keepdim=True).detach() > 0.0).type_as(feat_fix)

        feat_result = feat_init
        list_feat = []
        for _ in range(1, self.prop_time + 1):
            if self.args.preserve_input:
                feat_result = (1.0 - mask_fix) * feat_result + mask_fix * feat_fix
            feat_result = self._propagate_once(feat_result, offset, aff)
            list_feat.append(feat_result)

        return feat_result, list_feat, offset, aff, self.aff_scale_const.data
