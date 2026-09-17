# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A four-script research probe, not an application. It answers one question: can the
AOTAutograd **joint (forward+backward) graph** of the FLUX.2-small VAE decoder, driven by an
LOD-style wavelet loss on the latent, be rewritten so it contains **only operators that
ExecuTorch's WebGPU/Vulkan backend already implements** — i.e. with no `*_backward` ATen ops
left that would require new backward WGSL kernels?

`FLUX2_SMALL_LOD_WEBGPU_POC.md` is the write-up of the current answer (yes, validated to fp32
roundoff) and its open caveats. Keep it in sync with any change to the rewrites.

## Commands

```bash
# 1. Capture the raw joint graph and print its op inventory (FakeTensor; no real compute)
python flux2_lod_aot_probe.py --latent 32 --loss sym4        # --loss {sym4,l1,mse}, --dump-graph

# 2. Capture + rewrite, print the post-rewrite op inventory  -> flux2_lod_aot_ops_edge_normalized.txt
python flux2_targeted_rewrites.py

# 3. The test: numerical equivalence of rewritten vs. original graph
#    -> flux2_rewrite_validation.txt  (seconds; small 2x2 latent so the graph is executable)
python validate_flux2_rewrites.py

# 4. Same probe against the official checkpoint (needs diffusers + weights)
python flux2_real_aot_probe.py --model-id black-forest-labs/FLUX.2-small-decoder --latent 32
#    add --local-files-only for a cached checkpoint, --no-rewrite to see the raw graph
```

`validate_flux2_rewrites.py` is the only test. The two `.txt` files are checked-in outputs of
steps 2 and 3 — regenerate both whenever a rewrite changes.

Requires only `torch` (CPU is fine; everything is fp32/CPU or FakeTensor, so a CUDA capability
warning is irrelevant). `diffusers` is needed only for script 4; `executorch` is not installed
and nothing here calls it yet.

## Architecture

**Pipeline.** `flux2_lod_aot_probe.py` defines the model and captures the graph;
`flux2_targeted_rewrites.py` owns all graph surgery and exports `rewrite(gm)` + `opname(n)`;
`validate_flux2_rewrites.py` and `flux2_real_aot_probe.py` are both consumers of that single
`rewrite()`. There is no package — the probe module is loaded **by file path** via
`importlib`, so filenames are load-bearing.

**Capture.** `aot_export_module(m, (z, target), trace_joint=True, output_loss_index=0)`.
Two invariants follow from `trace_joint`:
- `forward` must return `(loss, <aux>.detach())` — non-loss outputs must be detached.
- Only `z` requires grad; every decoder parameter is frozen. That is what makes every
  backward node input-gradient-only (`output_mask == [True, False, False]`), which the
  rewrites **assert** and raise on. Unfreezing weights breaks `lower_conv_backward_input`
  and `lower_group_norm_backward_input` by design, not by accident.

**The model (`Flux2SmallDecoder`)** is a dependency-light reconstruction with random weights,
not a checkpoint load. Its purpose is *operator topology parity* with the Diffusers decoder —
accuracy is meaningless here, so evaluate edits by whether the op inventory still matches, not
by output quality. Parity details that look like bugs but are not:
- `UpDecoderBlock2D(layers=3)` is correct for `layers_per_block=2` (Diffusers uses N+1 resnets
  in the decoder).
- `VAEAttention` is written as explicit `bmm/softmax/bmm` rather than SDPA so AOTAutograd
  exposes primitive backward ops instead of a fused kernel. Script 4 calls
  `set_default_attn_processor()` for the same reason.
- x2 nearest upsampling is written as `view → expand → reshape`, not `interpolate`. This is
  the key trick: its backward is plain reductions instead of `arange/index/index_put`.

**The rewrites** (`rewrite()` applies them in order, then DCE + lint + recompile):
conv backward → transposed `aten.convolution` with statically computed `output_padding`;
GroupNorm input-grad → the explicit mean/rstd formula; softmax/sigmoid backward → closed form;
`sign` → `gt/lt` + cast + subtract; detach stripping; then layout normalization
(`view/_unsafe_view → view_copy`, `expand → expand_copy`, `t/transpose → permute`) to the
variants the WebGPU backend registers.

These rewrites are **shape-static**: they read `node.meta['val']` to compute `output_padding`
and the GroupNorm reshapes, and raise on missing/dynamic shape metadata. Dynamic shapes are
out of scope.

## What counts as passing

1. `flux2_targeted_rewrites.py` prints `remaining backward= []`.
2. `validate_flux2_rewrites.py` shows loss error exactly `0` and `grad_z` error ~1e-9.
3. The op inventory stays inside the WebGPU-supported set listed in the POC write-up — adding
   an op family to that list is a real result and needs the write-up updated, not just the
   `.txt`.

Rewrite counts (`conv_bwd=37, gn_bwd=30, softmax_bwd=1, sigmoid_bwd=29, sign=1`) are stable
invariants; the `detach` count is torch-version dependent (272 on 2.10, 817 on 2.6) and is not
a regression signal.
