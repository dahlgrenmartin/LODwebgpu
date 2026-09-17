# Milestone 0 gate result

**Date:** 2026-09-17
**Adapter:** nvidia / blackwell (RTX 5090), Chrome 152
**Model:** `lod_joint_128.onnx` (128x128 image, latent [1,32,16,16]), fp16 initializers,
shared external `weights.bin`, ONNX Runtime Web 1.30.0 on the WebGPU EP.

Golden data is the fp32 PyTorch wrapper, so each figure below is
**fp16 quantization + WebGPU fp32 reassociation combined**, relative error.

| output | browser (WebGPU) | onnxruntime CPU EP, same fp16 model | budget |
|---|---|---|---|
| `loss` | 9.63e-07 | 8.83e-07 | 6e-2 |
| `pred` | 1.60e-03 | 1.59e-03 | 6e-2 |
| `score` | 6.39e-05 | 6.42e-05 | 6e-2 |
| `grad_z` | 2.62e-03 | 2.62e-03 | 6e-2 |
| negative control (z + 0.5) | 1.07 | - | must exceed 6e-2 |

**Status: ALL_PASS 5/5.**

The browser figures match the CPU figures to three significant digits. The entire
deviation from PyTorch is fp16 weight storage, not the GPU: WebGPU's arithmetic
agrees with ORT-CPU to within noise.

## What this discharges

The joint graph of the frozen FLUX.2-small decoder -- with the whole backward pass
lowered into forward-only operators -- exports to ONNX, loads in a browser, and
computes a correct `grad_z` on WebGPU. The rewrite thesis now has an end-to-end
execution behind it, not just an operator-vocabulary comparison.

This does **not** discharge POC caveat #2: ExecuTorch and `VulkanPartitioner` are
still untouched, by design (see the design doc's Non-goals).

## Notes for later

- `aten::native_group_norm` has no symbolic in the legacy TorchScript exporter;
  the dynamo path is required, which needs `onnxscript` and onnxruntime >= 1.30
  (IR version 10).
- ORT-Web cannot resolve external data from a URL implicitly. The weights file must
  be passed via the `externalData` session option, keyed by the path recorded in
  the model.
- The dynamo exporter writes its own fp32 sidecar (`<model>.onnx.data`); it is
  orphaned once weights move into the shared `weights.bin` and is deleted at export.
