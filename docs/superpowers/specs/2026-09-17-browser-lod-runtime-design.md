# Browser LOD Runtime — Design

**Date:** 2026-09-17
**Status:** Approved design, not yet implemented
**Supersedes:** nothing. Discharges POC caveat #4 (`FLUX2_SMALL_LOD_WEBGPU_POC.md`).

## Context

`FLUX2_SMALL_LOD_WEBGPU_POC.md` established that the AOTAutograd joint graph of the
frozen FLUX.2-small VAE decoder, driven by a Sym4 wavelet detail loss on the latent,
can be rewritten so that no `*_backward` ATen op remains — every backward node lowers
into forward-style operators. As of 2026-09-17 that result is verified three ways:

- 6/6 Python smoke tests, including the real Diffusers `Decoder` class that
  `AutoencoderKLFlux2` uses;
- numerical equivalence to the original joint graph at 7.1e-9 max abs error on `grad_z`;
- 7/7 lowered kernels executed on real WebGPU hardware in Chrome 152, worst max abs
  error 7.6e-6.

What does not exist is a runtime. POC caveat #4 reads: *"LOD's level-3 wavelet detector
trace and 30-step Adam loop still need to be added to the browser runtime."* This
document designs that runtime.

## Goal

An **interactive demo**: the viewer supplies an image, watches the 30-step latent
optimization run client-side, and sees the resulting image and detector score.

Success is that it is convincing and works on a normal laptop. That makes runtime
maturity and load time first-class design constraints, and it rules out assuming a
discrete GPU.

## Non-goals

- **Discharging POC caveat #2 (ExecuTorch).** This build is evidence for the rewrite
  thesis, which is runtime-agnostic — not evidence about ExecuTorch specifically.
  `.pte` export and `VulkanPartitioner` stay a separate track.
- A reusable/embeddable library, a benchmark harness, or a score-parity harness. Those
  were considered and explicitly not chosen.
- Dynamic shapes. The rewrites are shape-static by construction; see §5.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Execution engine | ONNX Runtime Web, WebGPU EP | Every op in the final inventory has a standard ONNX equivalent *because* the rewrite eliminated the exotic ones. Mature, tuned kernels are the dominant variable for "works on a normal laptop"; neither a hand-written WGSL interpreter nor ExecuTorch offers that today. |
| Weight payload | fp16 initializers, fp32 compute | Halves download. Graph stays fp32 end-to-end, so every verified numerical result holds unchanged. |
| Display | GPU texture, no readback | Output buffer is bound directly as `var<storage, read>` in a fragment shader. Removes the only reason to throttle display frequency. |
| Feedback cadence | One step per `requestAnimationFrame`, canvas updated every step | Free once display is GPU-side; keeps the page responsive and self-paces to the adapter. |
| Latent init | VAE encoder, posterior mean | Faithful to the reference LOD implementation and converges fastest, at the cost of roughly doubling the download. |

## 1. Architecture & data flow

Two halves with a file boundary between them: an offline Python exporter, and a browser
runtime that knows nothing about PyTorch.

**Build time (Python, runs once).** Capture the joint graph → `rewrite(gm)` → wrap the
`GraphModule` in a thin `nn.Module` holding params and buffers as attributes, calling
`gm` in captured placeholder order. The wrapper is what gets exported, and it is
load-bearing: `aot_export_module` lifts every parameter into a graph *placeholder*, so
exporting `gm` directly yields an ONNX model with several hundred inputs. Wrapping turns
them into initializers and leaves exactly two real inputs, `z` and `target`.

Outputs: `lod_joint_<res>.onnx` (one per resolution, see §5), a shared external weights
file, `encoder.onnx`, and `manifest.json`.

The encoder is **not** produced by the rewrite, so it is a plain forward export and can
use dynamic shapes — one file covers both resolutions.

**Run time (browser).** ORT-Web with the WebGPU EP, constructed on *our* `GPUDevice` so
the render pass can bind ORT's output buffers directly. Per rAF tick:

```
z(GPUBuffer) ─┐
              ├─> ORT session ─> loss, grad_z, pred, detector_score   (all gpu-buffer)
target ───────┘                    │        │        │        │
                                   │        │        │        └─> staging ring ─> curve (lagged 2 frames)
                                   │        │        └─> render pass ─> canvas
                                   │        └─> Adam kernel (WGSL) ─> z' in place
                                   └─> staging ring
```

`z` and Adam's `m`/`v` are persistent `GPUBuffer`s that never leave the GPU across all
30 steps.

### Required change to the Python side

The graph's current non-loss output is `pred.mean().detach()` — a scalar. The browser
needs the image. `JointLOD.forward` must return three outputs:

```python
return loss, pred.detach(), detector_score.detach()
```

All are detached, which `trace_joint=True` already requires, so this does not disturb
the backward graph or invalidate any verified result. Note the graph has **four** outputs,
not three: `trace_joint` appends `grad_z` to whatever `forward` returns. The level-3 detector trace is
computed forward-only inside the graph and remains outside the backward pass, exactly as
it is today.

## 2. Components

### New Python (build-time)

| File | Responsibility |
|---|---|
| `export/wrap.py` | `nn.Module` wrapper holding params/buffers; calls `gm` in captured placeholder order. Converts placeholders to initializers. |
| `export/to_onnx.py` | ONNX export, fp16 initializers with `Cast`→fp32, shared external data, writes `manifest.json`. |
| `export/reference.py` | Dumps golden `z`/`target`/`loss`/`grad_z`/`pred`/`score` as raw binary for the browser parity test. |

### Modified Python

| File | Change |
|---|---|
| `flux2_lod_aot_probe.py` | `JointLOD.forward` returns `(loss, pred.detach(), score.detach())`. |
| `smoke_test.py` | Four new tests (§7). Existing six stay unchanged — they guard the rewrite. |

### New browser (TypeScript + WGSL)

Each owns exactly one GPU concern and is verifiable in isolation against a golden buffer.

| File | Responsibility |
|---|---|
| `session.ts` | Device creation; ORT-Web on the shared device; io-binding; `preferredOutputLocation`. |
| `adam.wgsl` / `adam.ts` | Adam over the latent; owns `m`, `v`, step count. |
| `display.wgsl` / `display.ts` | Fullscreen triangle; NCHW fp32 buffer → canvas. |
| `telemetry.ts` | Staging-buffer ring; scalar readback that never awaits the current frame. |
| `imageInput.ts` | Upload / drag-drop / bundled samples → resize → NCHW fp32 `target` buffer. |
| `latentInit.ts` | Encoder session; posterior mean; scaling/shift; produces `z₀`. |
| `app.ts` | rAF state machine (`idle → fetching → encoding → running(n) → done`); resolution selection. |
| `index.html` + styles | UI. |

## 3. Payload budget

| Item | Size |
|---|---|
| Joint decoder graph, fp16 initializers | ~56 MB |
| Encoder, fp16 | ~50–60 MB |
| ORT-Web bundle (wasm + WebGPU EP) | ~3–5 MB |
| Samples + app | < 1 MB |
| **Total first visit** | **~110–120 MB** |

fp16 initializers halve the *download*, which is the goal. ORT's constant folding will
materialize them back to fp32 at session init, so **GPU memory is unchanged** — load
time improves, footprint does not. This is a deliberate trade, not an oversight.

Cache API storage makes the second visit instant.

## 4. Model pipeline & the optimization loop

**Load & init (once per image).** Fetch both graphs; create both sessions on the shared
device. On image drop, run the encoder forward-only and take the posterior **mean** —
not a sample, so runs are reproducible — then apply the VAE `scaling_factor` /
`shift_factor` from the manifest to produce `z₀`.

**Destroy the encoder session before the loop starts.** It is dead weight for the next
30 steps, and freeing it returns its GPU memory exactly when the joint graph needs
headroom.

**Per rAF tick (×30), with no CPU sync in the chain:**

```
session.run({z, target}) → {loss, grad_z, pred, score}   all gpu-buffer
  → adam.wgsl(z, m, v, grad_z, step) → z in place
  → display.wgsl(pred) → canvas
  → telemetry: enqueue loss + score into staging ring, read frame n−2
```

Adam is textbook with bias correction. `lr`, `β₁`, `β₂`, `ε` and the step count live in
`manifest.json` so the Python reference and the browser cannot drift apart.

**These values must be read off the reference LOD implementation, which is not in this
repo.** They are not free parameters to invent — detector-score comparability depends on
matching them. Blocking input for Milestone 1.

At 256×256 the latent is `[1,32,32,32]` = 32,768 floats (one dispatch, one thread per
element). At 128×128 it is `[1,32,16,16]` = 8,192.

## 5. Resolution and static shapes

The lowerings are shape-static by design — `output_padding` and every `view` shape are
computed from `meta['val']`. There is therefore **no dynamic-shape ONNX**; each
resolution needs its own exported graph.

The weights are identical between resolutions, so the plan is to export 128×128 and
256×256 against a **shared external-data file**: download the weights once plus a small
per-resolution graph. This depends on both exports emitting identical initializer names
and offsets, which is verified in Milestone 0 rather than assumed.

**Fallback if sharing does not hold:** choose resolution from adapter limits *before*
downloading, and fetch only that graph.

Resolution is selected up front from adapter limits, with a timing check on step 1: if
the first step exceeds ~400 ms, the app offers to restart at 128×128 rather than
grinding through 30 slow frames.

## 6. Error handling

Guiding principle: **fail loudly and specifically rather than degrade silently.** A demo
that quietly falls back to something 50× slower reads as broken, not as unsupported.

| Failure | Handling |
|---|---|
| No WebGPU (`navigator.gpu` undefined, `requestAdapter()` → null) | Detect up front; state the requirement and the browser/flags needed. **Do not fall back to ORT's WASM EP** — 30 steps through 28M params there is minutes, not seconds. |
| Adapter limits too small | Check `maxStorageBufferBindingSize` / `maxBufferSize` before choosing resolution. Largest single activation at 256×256 is ~25 MB (96ch); weights ~112 MB fp32 after folding. Downgrade to 128×128, or refuse showing the actual numbers. |
| Device lost | Await `device.lost`; halt the loop; offer restart. Weights come from Cache API, so recovery costs no re-download. |
| Large fetch | Streamed fetch with real progress from `Content-Length`; retry on failure; Cache API storage. Private mode (no Cache API) proceeds uncached rather than failing. |
| OOM at session init | Catch; downgrade resolution; retry once; then stop with a clear message. |
| Divergence / NaN | Loss already arrives via the telemetry ring. If NaN, or rising for 3 consecutive steps, halt and report instead of rendering 30 frames of garbage. |
| Bad input image | `createImageBitmap` with `imageOrientation: 'from-image'` (EXIF), center-crop to square, resize, drop alpha. Reject non-decodable files plainly. |
| Tab backgrounded | rAF stops; loop pauses and resumes. The state machine treats this as normal. |

The two that decide whether this works in the wild are capability detection and making a
~110–120 MB download feel deliberate rather than broken.

## 7. Testing

### Python — extends `smoke_test.py`

Same assert-and-exit discipline. The existing six tests stay unchanged.

- **ONNX parity** — export, run under `onnxruntime` CPU EP, compare `loss`/`grad_z`/`pred`
  to the FX graph. Not bit-exact (fp32 reassociation differs), so it cannot use the `0.0`
  the in-graph tests use. Starting threshold: 1e-4 relative on `grad_z`, tightened once
  the real figure is measured.
- **fp16-initializer round trip** — the one place quality can silently degrade. Measured
  against the fp32 graph, not assumed.
- **Adam reference** — numpy implementation of the WGSL algorithm vs `torch.optim.Adam`
  over a fixed gradient sequence. Guards the `manifest.json` constants, the thing most
  likely to drift between Python and browser.
- **Shared external data** — verify both resolution exports reference one weights file at
  identical offsets.

### Browser — extends `webgpu_verify/`

Driven by chrome-devtools MCP, the same way the current 7/7 run was. The existing seven
kernel cases stay untouched.

- **End-to-end step parity** — load the real ONNX session, run one step against golden
  `z`/`target` from `export/reference.py`, compare to the Python reference bytes. This is
  the actual claim.
- **Adam kernel** — 30 steps on a golden gradient sequence vs the numpy reference.
- **Display shader** — render a known buffer, read the canvas back, compare pixels.
  Catches NCHW indexing and the `[-1,1]→[0,1]` mapping, which are easy to get subtly
  wrong and invisible by eye.
- **Every case carries a negative control.** The perturbation check run on 2026-09-17
  proved the harness discriminates rather than vacuously passing; this is a standing
  requirement, not a one-off.

### Convergence check (Python, once, before any UI)

With encoder init, confirm 30 steps measurably improves reconstruction and moves the
detector score across a handful of images. Not a unit test — it decides whether the demo
is convincing, and it is cheap to answer early.

### Manual gate

Run on at least one integrated GPU. Not meaningfully automatable; stated as a
requirement rather than pretended away.

## 8. Milestones

**Milestone 0 — de-risk before any UI exists.**

1. Add `pred` + score outputs; graph still lowers; 6/6 smoke tests still pass.
2. ONNX export succeeds from the wrapped module.
3. ORT-Web loads it in Chrome; one step matches Python within tolerance.
4. Shared external data across two resolutions verified.

If step 2 or 3 fails, the engine choice is wrong and the design is revisited before a
line of UI exists.

Milestone 0 is a **hard gate**: no work from Milestones 1–4 begins until all four steps
pass.

**Milestone 1 — headless loop.** Encoder init, 30 Adam steps, convergence check. No UI.

**Milestone 2 — display.** GPU-texture rendering, per-step canvas update, telemetry ring.

**Milestone 3 — app.** State machine, image input, resolution selection, error paths.

**Milestone 4 — verification.** Full browser parity suite with negative controls;
manual integrated-GPU gate.

## 9. Open risks

| Risk | Severity | Mitigation |
|---|---|---|
| ONNX export of a hand-mutated FX graph fails or produces an invalid model | High — invalidates the engine choice | Milestone 0, step 2, before anything else |
| ORT-Web op coverage gap for some op in the inventory | High | Milestone 0, step 3; fallback is a hand-written WGSL kernel for that one op |
| Shared external data across resolutions not achievable | Low | Documented fallback in §5 |
| 30 steps at 256×256 too slow on integrated GPUs | Medium | Resolution downgrade path already designed; measured at Milestone 1 |
| Encoder doubles payload beyond what a demo tolerates | Medium | Accepted deliberately; Cache API covers repeat visits |
| fp16 initializers measurably degrade output | Low | Explicit test in §7 |
| Adam hyperparameters unknown (reference LOD not in this repo) | High — blocks Milestone 1 | Must be supplied before the loop is built; see §4 |
