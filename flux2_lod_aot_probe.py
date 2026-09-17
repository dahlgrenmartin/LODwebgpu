#!/usr/bin/env python3
"""AOTAutograd operator probe for FLUX.2 small decoder + LOD-style loss.

This is deliberately dependency-light: it reconstructs the Diffusers Decoder
operator structure with PyTorch modules, so it can capture the joint forward /
backward graph without downloading model weights.  Channel widths match the
FLUX.2-small decoder: 96/192/384/384, latent_channels=32, layers_per_block=2.

The probe freezes all decoder parameters and requests gradients only w.r.t. z,
which matches LOD's latent optimization.
"""
from __future__ import annotations

import argparse
import collections
import operator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode


def silu_primitive(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ResnetBlock2D(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, cout, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(silu_primitive(self.norm1(x)))
        h = self.conv2(silu_primitive(self.norm2(h)))
        return h + (self.shortcut(x) if self.shortcut is not None else x)


class VAEAttention(nn.Module):
    """Single-head VAE attention matching Diffusers' deprecated VAE attn block shape.

    UNetMidBlock2D receives attention_head_dim=in_channels for Autoencoder Decoder,
    which gives one head.  The implementation is deliberately explicit
    bmm/softmax/bmm so AOTAutograd exposes the primitive backward graph.
    """
    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6, affine=True)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.Linear(channels, channels, bias=True)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        n, c, h, w = x.shape
        x = self.norm(x)
        x = x.reshape(n, c, h * w).transpose(1, 2)  # [N,T,C]
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        probs = scores.softmax(dim=-1)
        x = torch.bmm(probs, v)
        x = self.to_out(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x + residual


class MidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.resnet0 = ResnetBlock2D(channels, channels)
        self.attn = VAEAttention(channels)
        self.resnet1 = ResnetBlock2D(channels, channels)

    def forward(self, x):
        return self.resnet1(self.attn(self.resnet0(x)))


class UpDecoderBlock2D(nn.Module):
    def __init__(self, cin: int, cout: int, add_upsample: bool, layers: int = 3):
        super().__init__()
        blocks = []
        for i in range(layers):
            blocks.append(ResnetBlock2D(cin if i == 0 else cout, cout))
        self.resnets = nn.ModuleList(blocks)
        self.add_upsample = add_upsample
        self.upsample_conv = nn.Conv2d(cout, cout, 3, padding=1) if add_upsample else None

    def forward(self, x):
        for block in self.resnets:
            x = block(x)
        if self.add_upsample:
            n, c, h, w = x.shape
            x = x.view(n, c, h, 1, w, 1).expand(n, c, h, 2, w, 2).reshape(n, c, h * 2, w * 2).contiguous()
            x = self.upsample_conv(x)
        return x


class Flux2SmallDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        latent_channels = 32
        widths = [96, 192, 384, 384]
        rev = list(reversed(widths))  # 384,384,192,96
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.conv_in = nn.Conv2d(latent_channels, rev[0], 3, padding=1)
        self.mid = MidBlock(rev[0])
        self.up = nn.ModuleList([
            UpDecoderBlock2D(rev[0], rev[0], True),
            UpDecoderBlock2D(rev[0], rev[1], True),
            UpDecoderBlock2D(rev[1], rev[2], True),
            UpDecoderBlock2D(rev[2], rev[3], False),
        ])
        self.norm_out = nn.GroupNorm(32, widths[0], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(widths[0], 3, 3, padding=1)

    def forward(self, z):
        x = self.post_quant_conv(z)
        x = self.conv_in(x)
        x = self.mid(x)
        for block in self.up:
            x = block(x)
        x = self.conv_out(silu_primitive(self.norm_out(x)))
        return x


def zero_pad2d(x: torch.Tensor, left: int, right: int, top: int, bottom: int):
    """Zero-pad without aten.constant_pad_nd.

    constant_pad_nd's meta function does not propagate SymInts: a symbolic
    [1,3,H,W] comes out concrete, which specializes every downstream shape and
    breaks dynamic-shape capture.  cat with new_zeros keeps the symbols.
    """
    n, c, h, w = x.shape
    if left or right:
        x = torch.cat([x.new_zeros(n, c, h, left), x, x.new_zeros(n, c, h, right)], dim=3)
    n, c, h, w = x.shape
    if top or bottom:
        x = torch.cat([x.new_zeros(n, c, top, w), x, x.new_zeros(n, c, bottom, w)], dim=2)
    return x


# Symlet-4 analysis filters, exactly pywt.Wavelet('sym4').dec_lo / .dec_hi in
# decomposition order.  F.conv2d is cross-correlation while the DWT is a true
# convolution, so the assembled 2-D kernels are spatially reversed below.
SYM4_LO = torch.tensor([
    -0.07576571478927333, -0.02963552764599851,
     0.49761866763201545,  0.8037387518059161,
     0.29785779560527736, -0.09921954357684722,
    -0.012603967262037833, 0.0322231006040427,
], dtype=torch.float32)
SYM4_HI = torch.tensor([
    -0.0322231006040427, -0.012603967262037833,
     0.09921954357684722, 0.29785779560527736,
    -0.8037387518059161, 0.49761866763201545,
     0.02963552764599851, -0.07576571478927333,
], dtype=torch.float32)


class Sym4Level1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # Three level-1 detail filters. Apply independently to each RGB channel.
        kernels = torch.stack([
            torch.outer(SYM4_HI, SYM4_LO),
            torch.outer(SYM4_LO, SYM4_HI),
            torch.outer(SYM4_HI, SYM4_HI),
        ], dim=0)[:, None, :, :]  # [3,1,8,8]
        # repeat details per channel => grouped conv input C=3, output=9
        kernels = kernels.repeat(3, 1, 1, 1)
        # Reverse for cross-correlation; with pad=6 this reproduces
        # pywt.dwt2(..., mode="zero") coefficient-for-coefficient.
        kernels = kernels.flip(-1, -2).contiguous()
        self.register_buffer("kernels", kernels)

    def forward(self, residual):
        # Zero extension, pywt "zero" mode.  pad=6 (not 7) keeps the odd
        # samples of the full convolution, matching pywt's subsample phase.
        x = zero_pad2d(residual, 6, 6, 6, 6)
        d = F.conv2d(x, self.kernels, stride=2, groups=3)
        # sum/numel rather than mean(): mean's backward broadcasts the scalar
        # gradient with a materialised expand whose target shape is baked at
        # trace time, which breaks dynamic-shape capture.
        return d.abs().sum() / d.numel()


class Sym4Level3Detector(nn.Module):
    """Forward-only level-3 Sym4 residual-energy trace.

    Three successive level-1 transforms are applied to the previous
    approximation band, matching pywt.wavedec2(..., level=3). The returned
    scalar is the reference detector's L3 off-diagonal energy:
    0.5 * (mean(LH3^2) + mean(HL3^2)). The browser converts that energy to
    band-PSNR before fitting the 10-step slope.

    Band extraction uses view + select rather than arange/index_select on
    purpose: it keeps the operator inventory inside the vocabulary the rewrite
    already targets instead of introducing aten::arange and aten::index_select.
    """

    def __init__(self, channels: int = 3):
        super().__init__()
        bank = torch.stack([
            torch.outer(SYM4_LO, SYM4_LO),   # cA
            torch.outer(SYM4_HI, SYM4_LO),   # cH
            torch.outer(SYM4_LO, SYM4_HI),   # cV
            torch.outer(SYM4_HI, SYM4_HI),   # cD
        ], dim=0)[:, None, :, :]
        bank = bank.flip(-1, -2).contiguous()
        self.register_buffer("bank", bank)
        # Pre-repeated per-filter grouped-conv weights.  Building these here
        # rather than in forward keeps aten::repeat out of the exported graph.
        for name, k in (("w_a", 0), ("w_h", 1), ("w_v", 2), ("w_d", 3)):
            self.register_buffer(name, bank[k:k + 1].repeat(channels, 1, 1, 1).contiguous())
        self.channels = channels

    def bands(self, x, bank=None):
        """Return the three level-3 detail bands, each [N, C, h, w].

        One grouped conv per filter rather than a single 4-output conv plus a
        reshape: the reshape would bake the traced spatial size and break
        dynamic-shape capture, and this also skips the detail bands at levels 1
        and 2, which were previously computed and thrown away.
        """
        c = x.shape[1]
        if bank is None and c == self.channels:
            ws = (self.w_a, self.w_h, self.w_v, self.w_d)
        else:
            b = self.bank if bank is None else bank
            ws = tuple(b[k:k + 1].repeat(c, 1, 1, 1) for k in range(4))

        details = None
        for level in range(3):
            # Asymmetric (6,7): left pad 6 aligns the subsample phase with pywt,
            # right pad 7 gives floor((N+k-1)/2) coefficients for ODD N too.
            padded = zero_pad2d(x, 6, 7, 6, 7)
            if level == 2:
                details = [F.conv2d(padded, ws[k], stride=2, groups=c)
                           for k in (1, 2, 3)]
            x = F.conv2d(padded, ws[0], stride=2, groups=c)   # cA -> next level
        return details

    def forward(self, residual):
        h, v, _ = self.bands(residual)
        return 0.5 * ((h * h).mean() + (v * v).mean())


class JointLOD(nn.Module):
    def __init__(self, loss_kind: str):
        super().__init__()
        self.decoder = Flux2SmallDecoder()
        self.loss_kind = loss_kind
        self.sym4 = Sym4Level1Loss()
        self.detector = Sym4Level3Detector(channels=3)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    def forward(self, z, target):
        pred = self.decoder(z)
        residual = pred - target
        if self.loss_kind == "sym4":
            loss = self.sym4(residual)
        elif self.loss_kind == "l1":
            loss = residual.abs().mean()
        else:
            loss = residual.square().mean()
        # Non-loss outputs must be detached for trace_joint=True.  The detector
        # trace is forward-only and never enters the backward graph.
        score = self.detector(residual)
        return loss, pred.detach(), score.detach()


def opname(node):
    if node.op != "call_function":
        return None
    t = node.target
    if hasattr(t, "_schema"):
        return str(t._schema.name) + ("." + t._schema.overload_name if t._schema.overload_name else "")
    return str(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=32, help="latent spatial size; 32 => 256x256 image")
    ap.add_argument("--loss", choices=["sym4", "l1", "mse"], default="sym4")
    ap.add_argument("--dump-graph", action="store_true")
    args = ap.parse_args()

    m = JointLOD(args.loss).eval()
    total = sum(p.numel() for p in m.decoder.parameters())
    trainable = sum(p.numel() for p in m.decoder.parameters() if p.requires_grad)

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        z = torch.empty(1, 32, args.latent, args.latent, dtype=torch.float32, requires_grad=True)
        target = torch.empty(1, 3, args.latent * 8, args.latent * 8, dtype=torch.float32)
        gm, sig = aot_export_module(
            m, (z, target), trace_joint=True, output_loss_index=0
        )

    ops = [opname(n) for n in gm.graph.nodes]
    counts = collections.Counter(o for o in ops if o)

    print(f"torch={torch.__version__}")
    print(f"latent=[1,32,{args.latent},{args.latent}] output={args.latent*8}x{args.latent*8}")
    print(f"decoder_params={total:,} trainable_decoder_params={trainable:,}")
    print(f"loss={args.loss}")
    print("\nUNIQUE CALL_FUNCTION OPS")
    for name, count in sorted(counts.items()):
        print(f"{count:4d}  {name}")

    print("\nBACKWARD-SPECIFIC / IMPORTANT")
    keys = [
        "aten::convolution_backward", "aten::upsample_nearest2d_backward",
        "aten::native_group_norm_backward", "aten::_softmax_backward_data",
        "aten::bmm", "aten::mm", "aten::convolution", "aten::native_group_norm",
    ]
    for k in keys:
        found = [(n,c) for n,c in counts.items() if k in n]
        print(k, found or "-")

    print("\nBACKWARD SIGNATURE")
    print(sig.backward_signature)

    if args.dump_graph:
        print("\nJOINT GRAPH")
        print(gm.graph)


if __name__ == "__main__":
    main()