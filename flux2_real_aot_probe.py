#!/usr/bin/env python3
"""Probe the *real* FLUX.2 small decoder AOT joint graph.

Requires an environment with torch + diffusers and access to/cached weights.
This script intentionally uses an L1 residual loss: the decoder backward graph
is the target of this probe; the sym4 wavelet adds only a fixed pad/conv/abs
suffix, covered by flux2_lod_aot_probe.py.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flux2_targeted_rewrites import rewrite, opname  # noqa: E402

try:
    from diffusers import AutoencoderKLFlux2
except Exception as exc:
    raise SystemExit(f"diffusers with AutoencoderKLFlux2 is required: {exc}")


class RealDecoderJoint(nn.Module):
    def __init__(self, vae: AutoencoderKLFlux2):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z: torch.Tensor, target: torch.Tensor):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        pred = self.decoder(z)
        loss = (pred - target).abs().mean()
        return loss, pred.mean().detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.2-small-decoder")
    ap.add_argument("--latent", type=int, default=32, help="32 => 256x256 output")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--no-rewrite", action="store_true")
    args = ap.parse_args()

    vae = AutoencoderKLFlux2.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    ).eval()
    # Force the simple VAE attention processor where supported; it exposes
    # bmm/softmax primitives rather than depending on a fused kernel choice.
    try:
        vae.set_default_attn_processor()
    except Exception:
        pass

    joint = RealDecoderJoint(vae).eval()
    params = sum(p.numel() for p in joint.parameters())

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        z = torch.empty(1, vae.config.latent_channels, args.latent, args.latent,
                        dtype=torch.float32, requires_grad=True)
        target = torch.empty(1, vae.config.out_channels, args.latent * 8, args.latent * 8,
                             dtype=torch.float32)
        gm, sig = aot_export_module(joint, (z, target), trace_joint=True, output_loss_index=0)

    stats = None
    if not args.no_rewrite:
        stats = rewrite(gm)

    counts = collections.Counter(opname(n) for n in gm.graph.nodes if opname(n))
    print(f"torch={torch.__version__}")
    try:
        import diffusers
        print(f"diffusers={diffusers.__version__}")
    except Exception:
        pass
    print(f"model={args.model_id}")
    print(f"latent_channels={vae.config.latent_channels}")
    print(f"decoder_block_out_channels={getattr(vae.config, 'decoder_block_out_channels', None)}")
    print(f"block_out_channels={vae.config.block_out_channels}")
    print(f"decoder_params={params:,}")
    print(f"rewrites={stats}")
    print("\nOPS")
    for name, count in sorted(counts.items()):
        print(f"{count:4d}  {name}")
    print("\nremaining_backward=", [(n,c) for n,c in counts.items() if 'backward' in n])
    print("backward_signature=", sig.backward_signature)


if __name__ == "__main__":
    main()
