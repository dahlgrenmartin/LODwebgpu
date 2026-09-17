# Spike: fixed 1024x1024 ONNX on ORT-Web

**Date:** 2026-09-17
**Question:** Can a fixed 1024x1024 joint graph run in the browser, with the app
refusing any other size, as an alternative to shape-agnostic execution?

**Answer: no.** 1024x1024 fails at session creation. The practical ceiling on
ONNX Runtime Web is 768x768, and even that is too slow to be interactive.

## Measurements

Real FLUX.2-small decoder, fp16 initializers, ORT-Web 1.30 WebGPU EP,
Chrome 152, RTX 5090. One forward+backward pass, cold.

| image | session create | one step | loss | result |
|---|---|---|---|---|
| 256x256 | 1160 ms | 233 ms | 0.021969 | ok |
| 512x512 | 1105 ms | 694 ms | 0.010579 | ok |
| 768x768 | 1972 ms | 1574 ms | 0.006852 | ok |
| 1024x1024 | - | - | - | **memory access out of bounds** |

Export itself is not the problem: the 1024 graph exported in 23 s and is 1.3 MB
plus 63 MB of shared weights.

## Why it fails

The failure is at `InferenceSession.create`, before any inference, and it is a
WASM trap rather than a GPU error. The adapter reports
`maxStorageBufferBindingSize` and `maxBufferSize` of ~2 GB, so a 96-channel
1024x1024 activation (402 MB) is well within what the hardware allows. The wall
is ORT-Web's 32-bit WASM address space during graph planning, not the GPU.

Enabling cross-origin isolation (COOP/COEP) to unlock SharedArrayBuffer and the
multithreaded WASM build was tried and does not help; it addresses threading,
not address space.

## Second wall: time

Step cost scales roughly linearly with pixel count: 233 ms at 256, 694 ms at
512, 1574 ms at 768. Extrapolating, 1024x1024 would be ~2.8 s per step, so
30 steps is ~84 s even if the memory problem were solved. At 768x768 the full
run is ~47 s. Neither is an interactive demo.

Measured in the app (which also renders and runs Adam per step), 512x512 costs
1855 ms/step -- noticeably more than the 694 ms raw inference, so the per-step
scalar readback of loss and score is worth investigating before optimizing
anything else.

## What this means for the architecture

The spike was meant to test whether a fixed large size could sidestep the
shape-agnostic runtime. It cannot: ORT-Web caps out below the required
resolution, so "fix the size at 1024" is not a viable route regardless of the
no-resize/no-crop requirement.

That leaves the two routes from before, and this result weakens the ONNX one
further:

1. **WGSL interpreter** - shape-agnostic by construction, and not bound by a
   WASM heap because tensors live only in GPU buffers. The 2 GB binding limit
   the adapter reports is the real constraint, which 1024x1024 fits inside.
2. **Symbol materialization + ORT-Web** - even if the export problem were
   solved, this measurement says ORT-Web would still fail at 1024x1024.

## Current state

The demo is restored to 512x512 with exact-size enforcement: an image that is
not exactly 512x512 is refused with an explanation rather than resized or
cropped. Verified in Chrome: a 640x480 input is rejected, a 512x512 input runs
30/30 steps to loss 0.00274.
