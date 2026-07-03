import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvHead, ResizeConvBasicBlockNoCBAM, conv_relu

try:
    from ref_model.pvt import PVTTiny
except ModuleNotFoundError:
    PVTTiny = None


class PVTTinyFallback(nn.Module):
    """Dependency-light encoder fallback for smoke tests.

    Real retraining should use the original PVTTiny when timm/mmcv are
    installed. This fallback only preserves feature shapes/channels.
    """

    def __init__(self):
        super().__init__()
        self.fe2 = conv_relu(64, 64)
        self.fe3 = conv_relu(64, 128, stride=2)
        self.fe4 = conv_relu(128, 24, stride=2)
        self.fe5 = conv_relu(24, 48, stride=2)
        self.fe6 = conv_relu(48, 96, stride=2)
        self.fe7 = conv_relu(96, 192, stride=2)

    def forward(self, x):
        fe2 = self.fe2(x)
        fe3 = self.fe3(fe2)
        fe4 = self.fe4(fe3)
        fe5 = self.fe5(fe4)
        fe6 = self.fe6(fe5)
        fe7 = self.fe7(fe6)
        return fe2, fe3, fe4, fe5, fe6, fe7


class BackboneHWTiny(nn.Module):
    """Compiler-aligned CompletionFormerTiny backbone/decoder.

    This is intended for retraining a hardware-aligned checkpoint. It keeps the
    original PVTTiny encoder interface for now, while aligning the decoder and
    heads to the currently validated RHB/Host split:

    - ConvTranspose2d is replaced by bilinear resize + Conv2d.
    - Decoder BasicBlock removes BN/CBAM and uses a simple Conv/ReLU residual.
        - dec2 is represented as resize+conv+residual block; deployment splits
          the resize+up_conv by input channels while preserving this topology.
        - dep/gd dec1 use the validated full 96->32 head conv.
    - `_concat` uses the same host glue policy as the board runner by default.
    """

    def __init__(self, args, mode="rgbd", concat_mode="runner"):
        super().__init__()
        if mode != "rgbd":
            raise TypeError("BackboneHWTiny currently supports rgbd mode only")

        self.args = args
        self.mode = mode
        self.concat_mode = concat_mode
        self.num_neighbors = args.prop_kernel * args.prop_kernel - 1

        self.conv1_rgb = conv_relu(3, 48)
        self.conv1_dep = conv_relu(1, 16)
        self.conv1 = conv_relu(64, 64)

        if PVTTiny is None or getattr(args, "use_fallback_encoder", False):
            self.former = PVTTinyFallback()
        else:
            self.former = PVTTiny(
                in_chans=64,
                patch_size=2,
                pretrained=None,
                resnet_pretrained=not getattr(args, "from_scratch", True),
            )

        self.dec6 = ResizeConvBasicBlockNoCBAM(192, 96, (8, 8))
        self.dec5 = ResizeConvBasicBlockNoCBAM(96 + 96, 64, (16, 16))
        self.dec4 = ResizeConvBasicBlockNoCBAM(64 + 48, 48, (32, 32))
        self.dec3 = ResizeConvBasicBlockNoCBAM(48 + 24, 32, (64, 64))
        self.dec2 = ResizeConvBasicBlockNoCBAM(32 + 128, 32, (128, 128))

        self.dep_dec1 = ConvHead(32 + 64, 32, relu=True)
        self.dep_dec0 = ConvHead(32 + 64, 1, relu=True)
        self.gd_dec1 = ConvHead(32 + 64, 32, relu=True)
        self.gd_dec0 = ConvHead(32 + 64, self.num_neighbors, relu=False)

        self.conf_prop = getattr(args, "conf_prop", True)
        if self.conf_prop:
            self.cf_dec1 = ConvHead(32 + 64, 16, relu=True)
            self.cf_dec0 = ConvHead(16 + 64, 1, sigmoid=True)

    def _resize_like(self, fd, fe, mode=None):
        mode = mode or self.concat_mode
        if fd.shape[-2:] == fe.shape[-2:]:
            return fd
        if mode == "runner":
            # Matches the current board runner host glue for dec5/4/3/head.
            return F.interpolate(fd, size=fe.shape[-2:], mode="nearest")
        if mode == "bilinear":
            return F.interpolate(fd, size=fe.shape[-2:], mode="bilinear", align_corners=True)
        raise ValueError(f"Unsupported concat_mode: {mode}")

    def _concat(self, fd, fe, mode=None):
        return torch.cat((self._resize_like(fd, fe, mode=mode), fe), dim=1)

    def forward(self, rgb=None, depth=None):
        fe1 = self.conv1(torch.cat((self.conv1_rgb(rgb), self.conv1_dep(depth)), dim=1))
        fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1)

        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        # The opt2 board runner keeps this semantic block but executes the
        # resize+up_conv on RHB in two input-channel chunks from 64x64 inputs.
        fd2 = self.dec2(torch.cat((fd3, fe3), dim=1))

        head_in = self._concat(fd2, fe2)
        dep_fd1 = self.dep_dec1(head_in)
        init_depth = self.dep_dec0(self._concat(dep_fd1, fe1))

        gd_fd1 = self.gd_dec1(head_in)
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.conf_prop:
            cf_fd1 = self.cf_dec1(head_in)
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        return init_depth, guide, confidence
