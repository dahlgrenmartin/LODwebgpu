# FLUX.2 Small Decoder → LOD → ExecuTorch WebGPU probe

## What was actually run

Environment: PyTorch 2.10.0+cpu. `diffusers` and `executorch` are not installed in this runtime and network access is disabled, so the official checkpoint could not be loaded here.

To make the operator experiment executable anyway, `flux2_lod_aot_probe.py` reconstructs the Diffusers VAE decoder operator topology with the previously verified FLUX.2-small decoder widths `96/192/384/384`, `latent_channels=32`, `layers_per_block=2`, GroupNorm-32, SiLU, one VAE mid-attention block, three x2 nearest upsamplers and the post-quant 1x1 conv. Random weights are sufficient for an operator/topology probe.

The structural decoder has **27,947,235 parameters**, consistent with the ~28M small-decoder target. All decoder parameters are frozen. Only `z` receives a gradient.

The probe target is `z=[1,32,32,32]` → `256x256` fp32 output. A fixed Sym4 level-1 detail loss is included to exercise LOD's wavelet-gradient suffix. Its zero-extension boundary phase is operator-equivalent but not yet bit-for-bit PTWT parity. LOD's level-3 detector trace is detached from optimization and is not part of the backward graph.

## Raw AOTAutograd result

Before rewrites the joint graph contains:

- 37 `aten.convolution_backward` nodes, all input-gradient-only (`[True, False, False]`)
- 30 `aten.native_group_norm_backward` nodes, all input-gradient-only
- 1 `aten._softmax_backward_data`
- 29 SiLU/sigmoid backward paths
- nearest x2 upsampling
- one Sym4 detail-convolution backward

No gradients to decoder weights or biases are present.

## Rewrites implemented

`flux2_targeted_rewrites.py` performs these exact/static rewrites:

1. `convolution_backward(dInput only)` → `aten.convolution(..., transposed=True)` with statically computed `output_padding`.
2. GroupNorm input gradient → explicit `view/mul/sum/sub/div` formula using the forward mean/rstd and frozen affine weight.
3. Softmax backward → `(dy - sum(dy*y))*y`.
4. Sigmoid backward → `dy*y*(1-y)`.
5. `sign` from the L1 wavelet loss → `gt/lt` + fp32 cast + subtraction.
6. Diffusers' exact x2 nearest semantics are expressed as `view → expand → reshape`; backward therefore becomes reductions rather than `arange/index/index_put`.
7. Layout normalization maps `view/_unsafe_view → view_copy`, `expand → expand_copy`, and `t/transpose → permute`.
8. Explicit detach nodes are removed because autograd has already been captured into the joint graph.

After rewriting there are **no operator names containing `backward`**.

## Edge-normalized operator inventory

Ignoring Python tuple-unpack (`operator.getitem`), the final graph contains these ATen families:

```text
aten::_softmax
aten::_to_copy
aten::abs
aten::add.Scalar
aten::add.Tensor
aten::addmm
aten::bmm
aten::clone
aten::constant_pad_nd
aten::convolution
aten::div.Scalar
aten::expand_copy
aten::gt.Scalar
aten::lt.Scalar
aten::mean
aten::mm
aten::mul.Scalar
aten::mul.Tensor
aten::native_group_norm
aten::neg
aten::ones_like
aten::permute
aten::sigmoid
aten::sub.Tensor
aten::sum.dim_IntList
aten::unsqueeze
aten::view_copy
```

The exact counts are in `flux2_lod_aot_ops_edge_normalized.txt`.

## Numerical validation

`validate_flux2_rewrites.py` captures an executable small-spatial joint graph, applies every rewrite, and compares the rewritten outputs to the original AOTAutograd joint graph.

Observed result:

```text
loss:                  max abs error = 0
forward scalar:        max abs error = 0
grad_z [1,32,2,2]:     max abs error = 5.12227416e-09
                        mean abs error = 1.29848843e-09
```

This validates the rewrite algebra to fp32 roundoff on the full structural decoder graph, not just isolated formulas.

## WebGPU implication

Current ExecuTorch WebGPU already contains fp32 implementations for the important heavy operations in the normalized graph: normal/transposed convolution, native GroupNorm, softmax, BMM/MM, reductions, elementwise arithmetic, fill/ones, permute, view-copy/clone, expand-copy and padding.

The main result is therefore:

> For a 256x256 fp32 FLUX.2-small LOD PoC, the AOT backward can be rewritten into the existing WebGPU operator vocabulary. No new large backward WGSL kernel is obviously required by this operator probe.

This is stronger than the earlier assumption that general convolution/groupnorm backward kernels had to be added.

## What is not yet proven

1. The official `black-forest-labs/FLUX.2-small-decoder` checkpoint has not been loaded in this runtime. Use `flux2_real_aot_probe.py` in an environment with Diffusers and the checkpoint to confirm the exact production graph.
2. ExecuTorch is not installed here, so the rewritten joint FX graph has not yet been converted to an Edge `ExportedProgram`, partitioned by `VulkanPartitioner`, serialized to `.pte`, or executed through the browser WebGPU runtime.
3. The structural Sym4 suffix is not yet bit-for-bit PTWT boundary-compatible. That must be checked against the current LOD implementation before detector-score parity claims.
4. LOD's level-3 wavelet detector trace and 30-step Adam loop still need to be added to the browser runtime. The level-3 trace is forward-only; Adam acts only on the small latent tensor.
5. Browser peak memory and dispatch performance still need to be measured on the actual 256x256 joint `.pte` graph.

## Reproduce

Structural probe:

```bash
python flux2_lod_aot_probe.py --latent 32 --loss sym4
python flux2_targeted_rewrites.py
python validate_flux2_rewrites.py
```

Real official model probe (requires Diffusers + weights):

```bash
python flux2_real_aot_probe.py \
  --model-id black-forest-labs/FLUX.2-small-decoder \
  --latent 32
```

For a cached/offline checkpoint add `--local-files-only`.
re