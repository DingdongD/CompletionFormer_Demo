import torch
import torch.nn as nn

from .backbone_hw import BackboneHWTiny
from .nlspn_hw import NLSPNHW


class CompletionFormerHWTiny(nn.Module):
    """Training model for the current compiler-aligned Host/RHB design."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        if getattr(args, "model", "CompletionFormerTinyHW") not in (
            "CompletionFormerTinyHW",
            "CompletionFormerTiny",
        ):
            raise TypeError("CompletionFormerHWTiny only supports Tiny configuration")
        self.prop_time = getattr(args, "prop_time", 0)
        self.backbone = BackboneHWTiny(args, mode="rgbd")

        self.num_neighbors = args.prop_kernel * args.prop_kernel - 1
        if self.prop_time > 0:
            self.prop_layer = NLSPNHW(args, self.num_neighbors, 1, 3, args.prop_kernel)
        else:
            self.prop_layer = None

    def forward(self, sample):
        rgb = sample["rgb"]
        dep = sample["dep"]

        pred_init, guide, confidence = self.backbone(rgb, dep)
        pred_init = pred_init + dep

        y_inter = [pred_init]
        conf_inter = [confidence]
        if self.prop_layer is not None:
            y, y_inter, offset, aff, aff_const = self.prop_layer(pred_init, guide, confidence, dep, rgb)
        else:
            y = pred_init
            offset = torch.zeros_like(y)
            aff = torch.zeros_like(y)
            aff_const = torch.zeros_like(y).mean()

        y = torch.clamp(y, min=0)
        y_inter.reverse()
        conf_inter.reverse()
        if not getattr(self.args, "conf_prop", True):
            conf_inter = None

        return {
            "pred": y,
            "pred_init": pred_init,
            "pred_inter": y_inter,
            "guidance": guide,
            "offset": offset,
            "aff": aff,
            "gamma": aff_const,
            "confidence": conf_inter,
        }
