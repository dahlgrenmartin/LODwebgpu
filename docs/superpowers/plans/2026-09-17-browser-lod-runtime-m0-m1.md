# Browser LOD Runtime — Milestones 0–1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the rewritten FLUX.2-small joint graph to ONNX and run a verified 30-step Adam latent optimization headlessly in Chrome via ONNX Runtime Web on the WebGPU backend.

**Architecture:** An offline Python exporter wraps the rewritten FX `GraphModule` in an `nn.Module` so its lifted parameter placeholders become ONNX initializers, then emits one graph per resolution against a shared external weights file. A browser runtime creates the `GPUDevice`, hands it to ORT-Web, and drives one optimization step per tick with the latent and Adam moments resident on the GPU.

**Tech Stack:** PyTorch 2.12.0.dev (conda env `pytorch-5090`), diffusers 0.38, PyWavelets 1.8, onnx + onnxruntime (CPU EP, for parity tests), onnxruntime-web (WebGPU EP), TypeScript, WGSL, Vite, chrome-devtools MCP for browser verification.

**Spec:** `docs/superpowers/specs/2026-09-17-browser-lod-runtime-design.md`

## Global Constraints

- **Python is run through conda:** every command is `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python ...`. `conda run` **asserts on multi-line `-c` arguments**, so any multi-line snippet must go in a script file first. The base interpreter has torch 2.6 and will produce different `detach` counts.
- **ONNX opset 18** — required for `GroupNormalization`.
- **Compute is fp32 end to end.** fp16 appears only as initializer *storage*. No half-precision arithmetic anywhere.
- **Shape-static.** One joint graph per resolution: 256×256 (latent `[1,32,32,32]`, 32768 elements) and 128×128 (latent `[1,32,16,16]`, 8192 elements). The encoder is exported separately with dynamic shapes.
- **No WASM EP fallback.** If WebGPU is unavailable the runtime fails loudly.
- **Existing six tests in `smoke_test.py` must keep passing unchanged** — they guard the rewrite.
- **Numerical thresholds:** in-graph rewrite equivalence stays exact (`0.0` for loss, `1e-7` for `grad_z`). ONNX↔FX parity starts at `1e-4` relative on `grad_z`. WebGPU↔Python parity starts at `2e-5` max-abs, matching the existing `webgpu_verify` harness.
- **Adam hyperparameters have no defaults.** `lr`, `beta1`, `beta2`, `eps`, `steps` are required fields read from `manifest.json`; code raises if any is absent. Their real values must come from the reference LOD implementation, which is not in this repo. Tests use explicit fixture values.
- **Every browser verification case carries a negative control** — a deliberately perturbed input that must FAIL. A case without one is incomplete.

---

## File Structure

**Modified:**
- `flux2_lod_aot_probe.py` — add `Sym4Level3Detector`; `JointLOD.forward` returns three outputs.
- `flux2_targeted_rewrites.py` — repopulate `meta['val']` after rewriting.
- `smoke_test.py` — six new tests appended; existing six untouched.

**Created (Python):**
- `export/__init__.py`
- `export/wrap.py` — placeholder-order wrapper turning params into initializers.
- `export/to_onnx.py` — ONNX export, fp16 initializers, shared external data, manifest.
- `export/reference.py` — golden binary dumps for browser parity.
- `export/adam_reference.py` — numpy Adam used by both Python and browser tests.

**Created (browser):**
- `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`
- `web/src/manifest.ts` — typed manifest loader that rejects missing fields.
- `web/src/session.ts` — device creation, ORT-Web on the shared device, io-binding.
- `web/src/adam.wgsl`, `web/src/adam.ts` — Adam over the latent.
- `web/src/latentInit.ts` — encoder session, posterior mean, scaling.
- `web/src/harness.ts` — headless verification entry point; writes `window.__RESULTS__`.
- `web/index.html` — harness page (no product UI; that is Milestone 2).

---

# Milestone 0 — The Gate

### Task 1: Level-3 Sym4 detector

The level-1 loss exists; the level-3 detector trace does not. It is forward-only and feeds a detached output.

**Files:**
- Modify: `flux2_lod_aot_probe.py` (append after `Sym4Level1Loss`)
- Test: `smoke_test.py`

**Interfaces:**
- Consumes: `SYM4_LO`, `SYM4_HI` from `flux2_lod_aot_probe.py`
- Produces: `Sym4Level3Detector(channels: int = 3)`, `forward(x: Tensor[N,C,H,W]) -> Tensor[]` (scalar score); attribute `.bank` of shape `[4,1,8,8]` ordered `[cA, cH, cV, cD]`

- [ ] **Step 1: Write the failing test**

Append to `smoke_test.py`, above `def main()`:

```python
@test
def test_level3_detector_matches_pywt():
    """Level-3 Sym4 trace reproduces pywt.wavedec2(level=3) detail bands."""
    try:
        import pywt
        import numpy as np
    except ImportError:
        print("      SKIP (pywt not installed)")
        return
    import numpy as np

    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 1, 64, 64))
    det = probe.Sym4Level3Detector(channels=1)
    bands = det.bands(torch.tensor(x, dtype=torch.float64),
                      bank=det.bank.to(torch.float64))

    coeffs = pywt.wavedec2(x[0, 0], "sym4", mode="zero", level=3)
    cH3, cV3, cD3 = coeffs[1]
    for name, ref, got in (("cH3", cH3, bands[0]), ("cV3", cV3, bands[1]),
                           ("cD3", cD3, bands[2])):
        g = got[0, 0].numpy()
        assert g.shape == ref.shape, f"{name}: shape {g.shape} != {ref.shape}"
        err = np.abs(g - ref).max()
        assert err < 1e-6, f"{name}: max_err {err:.3e}"
        print(f"      {name}: shape={ref.shape} max_err={err:.2e}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k level3`
Expected: FAIL with `AttributeError: module 'probe' has no attribute 'Sym4Level3Detector'`

- [ ] **Step 3: Write minimal implementation**

Append to `flux2_lod_aot_probe.py` after `Sym4Level1Loss`:

```python
class Sym4Level3Detector(nn.Module):
    """Forward-only level-3 Sym4 detail trace.

    Three successive level-1 transforms, each applied to the previous
    approximation band, matching pywt.wavedec2(..., level=3).  The output is a
    scalar score; it is detached at the JointLOD boundary and never enters the
    backward graph.
    """

    def __init__(self, channels: int = 3):
        super().__init__()
        bank = torch.stack([
            torch.outer(SYM4_LO, SYM4_LO),   # cA
            torch.outer(SYM4_HI, SYM4_LO),   # cH
            torch.outer(SYM4_LO, SYM4_HI),   # cV
            torch.outer(SYM4_HI, SYM4_HI),   # cD
        ], dim=0)[:, None, :, :]
        self.register_buffer("bank", bank.flip(-1, -2).contiguous())
        self.channels = channels

    def bands(self, x, bank=None):
        """Return the three level-3 detail bands, each [N, C, h, w]."""
        b = self.bank if bank is None else bank
        c = x.shape[1]
        w = b.repeat(c, 1, 1, 1)          # [4C,1,8,8], group i -> channel i
        for _ in range(3):
            padded = F.pad(x, (6, 6, 6, 6), mode="constant", value=0.0)
            out = F.conv2d(padded, w, stride=2, groups=c)
            idx = torch.arange(0, 4 * c, 4, device=out.device)
            x = out.index_select(1, idx)                       # cA -> next level
        return [out.index_select(1, idx + k) for k in (1, 2, 3)]

    def forward(self, x):
        h, v, d = self.bands(x)
        return (h.abs().mean() + v.abs().mean() + d.abs().mean()) / 3.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k level3`
Expected: PASS, three bands printed with `max_err` around `1e-7`

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py`
Expected: `PASSED 7/7`

- [ ] **Step 6: Commit**

```bash
git add flux2_lod_aot_probe.py smoke_test.py
git commit -m "feat: add level-3 Sym4 detector trace, verified against pywt.wavedec2"
```

> **Note for the implementer:** the score reducer (mean of three band means) is a concrete, defensible definition, but the *exact* reducer used by the reference LOD implementation is unconfirmed. The band values are ground truth; only the final scalar may need adjusting. Flag this rather than assume.

---

### Task 2: JointLOD returns image and score

**Files:**
- Modify: `flux2_lod_aot_probe.py` (`JointLOD.__init__`, `JointLOD.forward`)
- Test: `smoke_test.py`

**Interfaces:**
- Consumes: `Sym4Level3Detector` from Task 1
- Produces: `JointLOD.forward(z, target) -> (loss, pred_detached, score_detached)`; the captured joint graph therefore has **four** outputs, `grad_z` appended last by `trace_joint`

- [ ] **Step 1: Write the failing test**

```python
@test
def test_joint_outputs_image_and_score():
    """Joint graph exposes loss, pred image, detector score, and grad_z."""
    m = probe.JointLOD("sym4").eval()
    with FakeTensorMode(allow_non_fake_inputs=True):
        z = torch.empty(1, 32, 8, 8, requires_grad=True)
        t = torch.empty(1, 3, 64, 64)
        gm, _ = aot_export_module(m, (z, t), trace_joint=True, output_loss_index=0)
    out_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    flat = out_node.args[0]
    assert len(flat) == 4, f"expected 4 graph outputs, got {len(flat)}"
    shapes = [tuple(n.meta["val"].shape) for n in flat]
    assert shapes[1] == (1, 3, 64, 64), f"pred shape {shapes[1]}"
    assert shapes[0] == () and shapes[2] == (), f"loss/score not scalar: {shapes}"
    assert shapes[3] == (1, 32, 8, 8), f"grad_z shape {shapes[3]}"
    rw.rewrite(gm)
    assert not _backward_ops(gm), f"backward survived: {_backward_ops(gm)}"
    print(f"      outputs: loss{shapes[0]} pred{shapes[1]} score{shapes[2]} grad_z{shapes[3]}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k joint_outputs`
Expected: FAIL with `expected 4 graph outputs, got 3`

- [ ] **Step 3: Write minimal implementation**

In `flux2_lod_aot_probe.py`, replace `JointLOD.__init__` and `forward`:

```python
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
        score = self.detector(pred)
        return loss, pred.detach(), score.detach()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k joint_outputs`
Expected: PASS

- [ ] **Step 5: Run full suite — the rewrite invariants must hold**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py`
Expected: `PASSED 8/8`. `test_structural_rewrite_eliminates_backward` still asserts `conv_bwd == 37`, `gn_bwd == 30`, `softmax_bwd == 1`, `sigmoid_bwd == 29`. If any changed, the detector leaked into the backward graph — stop and investigate rather than editing the expected counts.

- [ ] **Step 6: Regenerate artifacts and commit**

```bash
CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python flux2_targeted_rewrites.py > flux2_lod_aot_ops_edge_normalized.txt
CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py > smoke_test_results.txt
git add flux2_lod_aot_probe.py smoke_test.py flux2_lod_aot_ops_edge_normalized.txt smoke_test_results.txt
git commit -m "feat: JointLOD returns pred image and detector score as detached outputs"
```

---

### Task 3: Repopulate node metadata after rewriting

Rewritten nodes carry empty `meta`, which blocks any export path and silently poisons `_shape()`. This is a prerequisite for Task 5.

**Files:**
- Modify: `flux2_targeted_rewrites.py` (`rewrite`)
- Test: `smoke_test.py`

**Interfaces:**
- Produces: `rewrite(gm, example_args=None)` — when `example_args` is supplied, every `call_function` node has `meta['val']` populated on return

- [ ] **Step 1: Write the failing test**

```python
@test
def test_rewrite_repopulates_meta():
    """Every rewritten node carries meta['val'] so export can consume the graph."""
    m = probe.JointLOD("sym4").eval()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        z = torch.empty(1, 32, 4, 4, requires_grad=True)
        t = torch.empty(1, 3, 32, 32)
        gm, sig = aot_export_module(m, (z, t), trace_joint=True, output_loss_index=0)
    rw.rewrite(gm, example_args=_placeholder_args(gm, sig, m, z, t, mode))
    missing = [n.name for n in gm.graph.nodes
               if n.op == "call_function" and "val" not in n.meta]
    assert not missing, f"{len(missing)} nodes without meta['val']: {missing[:5]}"
    print(f"      all {sum(1 for n in gm.graph.nodes if n.op=='call_function')} nodes have meta['val']")
```

Add this helper next to `_counts` in `smoke_test.py` — Task 5 reuses it:

```python
def _placeholder_args(gm, sig, module, z, target, mode=None):
    """Build the positional arg list for a lifted joint graph, in placeholder order."""
    state = module.state_dict()
    umap = {sig.user_inputs[0]: z, sig.user_inputs[1]: target}
    args = []
    for n in gm.graph.nodes:
        if n.op != "placeholder":
            continue
        if n.name in sig.inputs_to_parameters:
            t = state[sig.inputs_to_parameters[n.name]]
        elif n.name in sig.inputs_to_buffers:
            t = state[sig.inputs_to_buffers[n.name]]
        elif n.name in umap:
            t = umap[n.name]
        else:
            raise KeyError(n.name)
        args.append(mode.from_tensor(t) if mode is not None and not isinstance(
            t, torch._subclasses.fake_tensor.FakeTensor) else t)
    return args
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k repopulates_meta`
Expected: FAIL — either `TypeError: rewrite() got an unexpected keyword argument 'example_args'`, or a long list of nodes without `meta['val']`

- [ ] **Step 3: Write minimal implementation**

In `flux2_targeted_rewrites.py`, replace `rewrite`:

```python
def rewrite(gm, example_args=None):
    stats={}
    stats['conv_bwd']=lower_conv_backward_input(gm)
    stats['gn_bwd']=lower_group_norm_backward_input(gm)
    stats['softmax_bwd']=lower_softmax_backward(gm)
    stats['sigmoid_bwd']=lower_sigmoid_backward(gm)
    stats['sign']=lower_sign(gm)
    stats['detach']=strip_detach(gm)
    stats['layout']=normalize_layout_ops(gm)
    gm.graph.eliminate_dead_code(); gm.graph.lint(); gm.recompile()
    if example_args is not None:
        # Nodes created by the passes above carry empty meta.  Export paths and
        # _shape() both require meta['val'], so re-propagate it over the whole graph.
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp
        from torch._subclasses.fake_tensor import FakeTensorMode
        mode = next((a.fake_mode for a in example_args
                     if hasattr(a, 'fake_mode') and a.fake_mode is not None), None)
        FakeTensorProp(gm, mode=mode or FakeTensorMode(allow_non_fake_inputs=True)
                       ).propagate(*example_args)
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k repopulates_meta`
Expected: PASS

- [ ] **Step 5: Confirm the default path is unchanged**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py`
Expected: `PASSED 9/9`. `example_args` defaults to `None`, so every existing caller behaves exactly as before.

- [ ] **Step 6: Commit**

```bash
git add flux2_targeted_rewrites.py smoke_test.py
git commit -m "feat: optionally repopulate meta['val'] after rewrite via FakeTensorProp"
```

---

### Task 4: Export wrapper

`aot_export_module` lifts every parameter into a placeholder. Exporting `gm` directly yields hundreds of ONNX inputs; wrapping converts them to initializers.

**Files:**
- Create: `export/__init__.py` (empty), `export/wrap.py`
- Test: `smoke_test.py`

**Interfaces:**
- Produces: `ExportWrapper(gm, sig, module)` with `forward(z, target) -> (loss, pred, score, grad_z)`

- [ ] **Step 1: Write the failing test**

```python
@test
def test_export_wrapper_matches_graph():
    """Wrapper with 2 inputs reproduces the lifted graph's outputs exactly."""
    import sys
    sys.path.insert(0, str(ROOT))
    from export.wrap import ExportWrapper

    torch.manual_seed(11)
    m = probe.JointLOD("sym4").eval()
    z = torch.randn(1, 32, 2, 2, requires_grad=True)
    t = torch.randn(1, 3, 16, 16)
    gm, sig = aot_export_module(m, (z, t), trace_joint=True, output_loss_index=0)
    args = _placeholder_args(gm, sig, m, z.detach(), t)
    with torch.no_grad():
        ref = gm(*args)
        got = ExportWrapper(gm, sig, m)(z.detach(), t)
    assert len(ref) == len(got) == 4, f"{len(ref)} vs {len(got)}"
    for i, (a, b) in enumerate(zip(ref, got)):
        err = (a - b).abs().max().item()
        assert err == 0.0, f"output[{i}] differs by {err:.3e}"
    print(f"      4 outputs identical; wrapper exposes {2} inputs")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k export_wrapper`
Expected: FAIL with `ModuleNotFoundError: No module named 'export'`

- [ ] **Step 3: Write minimal implementation**

Create `export/__init__.py` (empty file). Create `export/wrap.py`:

```python
"""Wrap a lifted joint GraphModule so ONNX sees initializers, not placeholders."""
from __future__ import annotations

import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """Exposes forward(z, target); every parameter/buffer becomes an initializer.

    aot_export_module lifts parameters into graph placeholders.  Exporting the
    GraphModule directly would therefore produce an ONNX model with one input per
    parameter.  Registering them here as buffers makes the exporter fold them into
    initializers instead, leaving exactly two real inputs.
    """

    def __init__(self, gm, sig, module: nn.Module):
        super().__init__()
        self.gm = gm
        state = module.state_dict()
        self._plan: list[tuple[str, str]] = []   # (kind, key)
        for n in gm.graph.nodes:
            if n.op != "placeholder":
                continue
            if n.name in sig.inputs_to_parameters:
                key = self._store(state[sig.inputs_to_parameters[n.name]], n.name)
                self._plan.append(("const", key))
            elif n.name in sig.inputs_to_buffers:
                key = self._store(state[sig.inputs_to_buffers[n.name]], n.name)
                self._plan.append(("const", key))
            elif n.name == sig.user_inputs[0]:
                self._plan.append(("z", ""))
            elif n.name == sig.user_inputs[1]:
                self._plan.append(("target", ""))
            else:
                raise KeyError(f"unclassified placeholder {n.name}")

    def _store(self, tensor: torch.Tensor, name: str) -> str:
        key = "c_" + name.replace(".", "_")
        self.register_buffer(key, tensor.detach().clone(), persistent=True)
        return key

    def forward(self, z, target):
        args = []
        for kind, key in self._plan:
            if kind == "z":
                args.append(z)
            elif kind == "target":
                args.append(target)
            else:
                args.append(getattr(self, key))
        return self.gm(*args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k export_wrapper`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add export/__init__.py export/wrap.py smoke_test.py
git commit -m "feat: ExportWrapper converts lifted parameter placeholders to initializers"
```

---

### Task 5: ONNX export with CPU parity

This is the task that can invalidate the engine choice. Do not proceed past it on a failure — escalate.

**Files:**
- Create: `export/to_onnx.py`
- Test: `smoke_test.py`

**Interfaces:**
- Produces: `export_joint(res: int, out_dir: Path, fp16: bool = False, external_data: bool = False, seed: int = 0) -> Path`, returning the written `.onnx` path; output names are exactly `["loss", "pred", "score", "grad_z"]`, input names `["z", "target"]`

- [ ] **Step 1: Write the failing test**

```python
@test
def test_onnx_matches_fx_graph():
    """Exported ONNX reproduces the FX graph under onnxruntime CPU EP."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("      SKIP (onnxruntime not installed)")
        return
    import sys, tempfile
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from export.to_onnx import export_joint, build_graph

    with tempfile.TemporaryDirectory() as td:
        path = export_joint(res=32, out_dir=Path(td), seed=5)
        gm, wrapper, z, t = build_graph(res=32, seed=5)
        with torch.no_grad():
            ref = wrapper(z, t)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"z": z.numpy(), "target": t.numpy()})

    names = ["loss", "pred", "score", "grad_z"]
    for i, name in enumerate(names):
        a = ref[i].numpy() if ref[i].ndim else np.array(ref[i].item())
        b = np.asarray(got[i])
        denom = max(1e-6, float(np.abs(a).max()))
        rel = float(np.abs(a - b).max()) / denom
        assert rel < 1e-4, f"{name}: rel err {rel:.3e}"
        print(f"      {name}: rel_err={rel:.2e} shape={b.shape}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k onnx_matches`
Expected: FAIL with `ModuleNotFoundError: No module named 'export.to_onnx'`

- [ ] **Step 3: Write minimal implementation**

Create `export/to_onnx.py`:

```python
"""Export the rewritten joint graph to ONNX."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode

ROOT = Path(__file__).resolve().parent.parent

def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

probe = _load("probe", "flux2_lod_aot_probe.py")
rw = _load("rw", "flux2_targeted_rewrites.py")

from export.wrap import ExportWrapper   # noqa: E402

OUTPUT_NAMES = ["loss", "pred", "score", "grad_z"]
INPUT_NAMES = ["z", "target"]


def build_graph(res: int, seed: int = 0):
    """Return (gm, wrapper, z, target) for a latent of spatial size `res`."""
    torch.manual_seed(seed)
    m = probe.JointLOD("sym4").eval()
    z = torch.randn(1, 32, res, res)
    t = torch.randn(1, 3, res * 8, res * 8)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        fz = torch.empty(1, 32, res, res, requires_grad=True)
        ft = torch.empty(1, 3, res * 8, res * 8)
        gm, sig = aot_export_module(m, (fz, ft), trace_joint=True, output_loss_index=0)
    rw.rewrite(gm, example_args=None)
    wrapper = ExportWrapper(gm, sig, m).eval()
    return gm, wrapper, z, t


def export_joint(res: int, out_dir: Path, fp16: bool = False,
                 external_data: bool = False, seed: int = 0) -> Path:
    gm, wrapper, z, t = build_graph(res, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lod_joint_{res * 8}.onnx"
    torch.onnx.export(
        wrapper, (z, t), str(path),
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        opset_version=18, dynamo=False, do_constant_folding=False,
    )
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k onnx_matches`
Expected: PASS

**If it fails, this is the gate firing.** Work the decision tree in order, and record which branch was taken:
1. TorchScript exporter rejects the `GraphModule` → retry with `dynamo=True`, passing `example_args` to `rw.rewrite` in `build_graph` so `meta['val']` is populated (Task 3 exists for this).
2. A specific aten op has no ONNX symbolic → note the op name; the fallback is a custom symbolic for that one op, not abandoning the approach.
3. Export succeeds but parity fails beyond `1e-4` → compare per-output; a single diverging output localizes the bad lowering.
4. Multiple unrelated ops unsupported → the engine choice is wrong. **Stop and escalate to the design author before writing further code.**

- [ ] **Step 5: Commit**

```bash
git add export/to_onnx.py smoke_test.py
git commit -m "feat: export rewritten joint graph to ONNX with CPU parity test"
```

---

### Task 6: fp16 initializers and shared external data

**Files:**
- Modify: `export/to_onnx.py`
- Test: `smoke_test.py`

**Interfaces:**
- Produces: `export_joint(..., fp16=True, external_data=True)`; writes `weights.bin` alongside; `write_manifest(out_dir, resolutions, adam) -> Path`

- [ ] **Step 1: Write the failing test**

```python
@test
def test_fp16_initializers_and_shared_external_data():
    """fp16 storage keeps parity, and both resolutions share one weights file."""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("      SKIP (onnx/onnxruntime not installed)")
        return
    import sys, tempfile
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from export.to_onnx import export_joint, build_graph

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        p32 = export_joint(res=32, out_dir=d, fp16=True, external_data=True, seed=5)
        p16 = export_joint(res=16, out_dir=d, fp16=True, external_data=True, seed=5)
        assert (d / "weights.bin").exists(), "shared weights.bin not written"
        sizes = sorted(f.stat().st_size for f in d.glob("*.onnx"))
        assert sizes[-1] < 5 * 1024 * 1024, f"graph file too large: {sizes[-1]}"

        def initializers(p):
            m = onnx.load(str(p), load_external_data=False)
            return {i.name: (i.data_type, i.raw_data[:0]) for i in m.graph.initializer}
        a, b = initializers(p32), initializers(p16)
        shared = set(a) & set(b)
        assert shared, "no initializer names shared between resolutions"
        for n in shared:
            assert a[n][0] == b[n][0], f"{n}: dtype differs between resolutions"
        print(f"      {len(shared)} initializers shared; graphs {sizes} bytes")

        _, wrapper, z, t = build_graph(res=32, seed=5)
        with torch.no_grad():
            ref = wrapper(z, t)
        sess = ort.InferenceSession(str(p32), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"z": z.numpy(), "target": t.numpy()})
        g = np.asarray(got[3])
        r = ref[3].numpy()
        rel = float(np.abs(r - g).max()) / max(1e-6, float(np.abs(r).max()))
        assert rel < 1e-3, f"fp16 initializers degraded grad_z: rel {rel:.3e}"
        print(f"      fp16 grad_z rel_err={rel:.2e}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k fp16_initializers`
Expected: FAIL with `shared weights.bin not written`

- [ ] **Step 3: Write minimal implementation**

Append to `export/to_onnx.py` and route `export_joint` through it:

```python
import json
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import convert_model_to_external_data


def _to_fp16_initializers(model: onnx.ModelProto) -> None:
    """Store initializers as fp16 and insert Cast->fp32 so compute stays fp32."""
    import numpy as np
    graph = model.graph
    casts = []
    for init in graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        half = numpy_helper.from_array(arr, init.name + "_fp16")
        init.CopyFrom(half)
        casts.append(init.name)
    for name in casts:
        out = name + "_fp32"
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == name + "_fp16" or inp == name:
                    node.input[i] = out
        graph.node.insert(0, onnx.helper.make_node(
            "Cast", [name + "_fp16"], [out], to=onnx.TensorProto.FLOAT))


def finalize(path: Path, fp16: bool, external_data: bool) -> None:
    model = onnx.load(str(path))
    if fp16:
        _to_fp16_initializers(model)
    if external_data:
        convert_model_to_external_data(
            model, all_tensors_to_one_file=True, location="weights.bin",
            size_threshold=1024, convert_attribute=False)
    onnx.save(model, str(path))


def write_manifest(out_dir: Path, resolutions: list[int], adam: dict) -> Path:
    for key in ("lr", "beta1", "beta2", "eps", "steps"):
        if key not in adam:
            raise KeyError(f"adam.{key} is required and has no default")
    manifest = {
        "resolutions": [
            {"image": r * 8, "latent": [1, 32, r, r], "numel": 32 * r * r,
             "graph": f"lod_joint_{r * 8}.onnx"} for r in resolutions
        ],
        "weights": "weights.bin",
        "encoder": "encoder.onnx",
        "adam": adam,
        "outputs": OUTPUT_NAMES,
        "inputs": INPUT_NAMES,
    }
    p = Path(out_dir) / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p
```

Then in `export_joint`, after `torch.onnx.export(...)`:

```python
    if fp16 or external_data:
        finalize(path, fp16=fp16, external_data=external_data)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k fp16_initializers`
Expected: PASS

If the two resolutions do not share initializer names or offsets, apply the fallback from spec §5: pick resolution from adapter limits before download and fetch one graph only. Record the decision in the spec.

- [ ] **Step 5: Commit**

```bash
git add export/to_onnx.py smoke_test.py
git commit -m "feat: fp16 initializers, shared external data, manifest writer"
```

---

### Task 7: Golden reference dump

**Files:**
- Create: `export/reference.py`
- Test: `smoke_test.py`

**Interfaces:**
- Produces: `dump_reference(res: int, out_dir: Path, seed: int = 5) -> Path` writing `reference_<image>.bin` plus `reference_<image>.json` describing `{name: {offset, count, shape}}`, dtype float32 little-endian throughout

- [ ] **Step 1: Write the failing test**

```python
@test
def test_reference_dump_roundtrips():
    """Golden binary matches the wrapper outputs it was produced from."""
    import sys, tempfile, json
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from export.reference import dump_reference
    from export.to_onnx import build_graph

    with tempfile.TemporaryDirectory() as td:
        blob = dump_reference(res=16, out_dir=Path(td), seed=5)
        meta = json.loads(Path(str(blob).replace(".bin", ".json")).read_text())
        raw = np.frombuffer(Path(blob).read_bytes(), dtype="<f4")
        _, wrapper, z, t = build_graph(res=16, seed=5)
        with torch.no_grad():
            ref = wrapper(z, t)
        named = {"z": z, "target": t, "loss": ref[0], "pred": ref[1],
                 "score": ref[2], "grad_z": ref[3]}
        for name, tensor in named.items():
            e = meta[name]
            got = raw[e["offset"]:e["offset"] + e["count"]]
            want = tensor.detach().numpy().ravel().astype("<f4")
            assert got.shape == want.shape, f"{name}: {got.shape} vs {want.shape}"
            assert np.array_equal(got, want), f"{name}: bytes differ"
            print(f"      {name}: {e['count']} values @ {e['offset']}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k reference_dump`
Expected: FAIL with `ModuleNotFoundError: No module named 'export.reference'`

- [ ] **Step 3: Write minimal implementation**

Create `export/reference.py`:

```python
"""Dump golden tensors for the browser parity harness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from export.to_onnx import build_graph


def dump_reference(res: int, out_dir: Path, seed: int = 5) -> Path:
    _, wrapper, z, t = build_graph(res=res, seed=seed)
    with torch.no_grad():
        loss, pred, score, grad_z = wrapper(z, t)

    named = {"z": z, "target": t, "loss": loss, "pred": pred,
             "score": score, "grad_z": grad_z}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = out_dir / f"reference_{res * 8}.bin"

    meta, chunks, offset = {}, [], 0
    for name, tensor in named.items():
        arr = tensor.detach().numpy().ravel().astype("<f4")
        meta[name] = {"offset": offset, "count": int(arr.size),
                      "shape": list(tensor.shape)}
        chunks.append(arr)
        offset += int(arr.size)

    blob.write_bytes(np.concatenate(chunks).tobytes())
    Path(str(blob).replace(".bin", ".json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return blob
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k reference_dump`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add export/reference.py smoke_test.py
git commit -m "feat: golden reference dump for browser parity"
```

---

### Task 8: Browser ONNX parity — THE GATE

**Files:**
- Create: `export/make_fixtures.py`, `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`, `web/index.html`, `web/src/manifest.ts`, `web/src/session.ts`, `web/src/harness.ts`
- Test: driven via chrome-devtools MCP against the running dev server

**Interfaces:**
- Consumes: `lod_joint_128.onnx`, `weights.bin`, `manifest.json`, `reference_128.bin/.json` from Tasks 5–7
- Produces: `createSession(manifest, res) -> {device, session, run(z, target)}`; `window.__RESULTS__ = {status, cases[]}`; `window.__DONE__ = true`

- [ ] **Step 1: Generate the fixtures the page will load**

`conda run` asserts on multi-line `python -c` arguments, so fixture generation lives in a
committed script rather than inline. Create `export/make_fixtures.py`:

```python
#!/usr/bin/env python3
"""Write browser harness fixtures into web/public/models.

    python export/make_fixtures.py m0
    python export/make_fixtures.py adam
    python export/make_fixtures.py encoder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "web" / "public" / "models"

# Fixture hyperparameters.  NOT the reference LOD values -- see Global
# Constraints.  They exist so manifest validation passes.
FIXTURE_ADAM = {"lr": 0.05, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "steps": 30}
SEED = 5
RES = 16   # latent 16 -> 128x128 image


def stage_m0() -> None:
    from export.to_onnx import export_joint, write_manifest
    from export.reference import dump_reference
    OUT.mkdir(parents=True, exist_ok=True)
    export_joint(res=RES, out_dir=OUT, fp16=True, external_data=True, seed=SEED)
    dump_reference(res=RES, out_dir=OUT, seed=SEED)
    write_manifest(OUT, [RES], FIXTURE_ADAM)
    print(f"m0 fixtures -> {OUT}")


def stage_adam() -> None:
    import numpy as np
    from export.adam_reference import adam_steps
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    n = 32 * RES * RES
    steps = FIXTURE_ADAM["steps"]
    z0 = rng.standard_normal(n).astype(np.float32)
    grads = [rng.standard_normal(n).astype(np.float32) for _ in range(steps)]
    cfg = {k: FIXTURE_ADAM[k] for k in ("lr", "beta1", "beta2", "eps")}
    expected = adam_steps(z0.copy(), grads, **cfg)
    (OUT / "adam_golden.json").write_text(json.dumps({
        "z0": z0.tolist(), "grads": [g.tolist() for g in grads],
        "expected": expected.tolist(), "cfg": cfg,
    }))
    print(f"adam golden -> {OUT / 'adam_golden.json'}")


def stage_encoder() -> None:
    from export.to_onnx import export_encoder
    OUT.mkdir(parents=True, exist_ok=True)
    export_encoder(OUT)
    print(f"encoder -> {OUT / 'encoder.onnx'}")


STAGES = {"m0": stage_m0, "adam": stage_adam, "encoder": stage_encoder}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES) + ["all"])
    args = ap.parse_args()
    for name in (sorted(STAGES) if args.stage == "all" else [args.stage]):
        STAGES[name]()
```

Run it:

```bash
cd D:/LODwebgpu
CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python export/make_fixtures.py m0
```

The Adam values are **test fixtures**, not the real hyperparameters. They exist so the manifest validates; they must be replaced before Milestone 1 completes.

- [ ] **Step 2: Scaffold the web project**

```bash
cd D:/LODwebgpu/web
npm init -y
npm install --save onnxruntime-web
npm install --save-dev vite typescript @webgpu/types
```

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022", "module": "ESNext", "moduleResolution": "bundler",
    "strict": true, "skipLibCheck": true, "types": ["@webgpu/types"],
    "lib": ["ES2022", "DOM"]
  },
  "include": ["src"]
}
```

`web/vite.config.ts`:

```ts
import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 8732 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
```

- [ ] **Step 3: Write the manifest loader and session**

`web/src/manifest.ts`:

```ts
export interface AdamConfig {
  lr: number; beta1: number; beta2: number; eps: number; steps: number;
}
export interface Resolution {
  image: number; latent: number[]; numel: number; graph: string;
}
export interface Manifest {
  resolutions: Resolution[]; weights: string; encoder: string;
  adam: AdamConfig; inputs: string[]; outputs: string[];
}

const ADAM_KEYS: (keyof AdamConfig)[] = ['lr', 'beta1', 'beta2', 'eps', 'steps'];

export async function loadManifest(url: string): Promise<Manifest> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`manifest fetch failed: ${res.status}`);
  const m = (await res.json()) as Manifest;
  if (!m.adam) throw new Error('manifest.adam missing');
  for (const k of ADAM_KEYS) {
    if (typeof m.adam[k] !== 'number') {
      throw new Error(`manifest.adam.${k} is required and has no default`);
    }
  }
  if (!m.resolutions?.length) throw new Error('manifest.resolutions empty');
  return m;
}
```

`web/src/session.ts`:

```ts
import * as ort from 'onnxruntime-web/webgpu';
import type { Manifest, Resolution } from './manifest';

export interface Runner {
  device: GPUDevice;
  session: ort.InferenceSession;
  resolution: Resolution;
  run(z: ort.Tensor, target: ort.Tensor): Promise<Record<string, ort.Tensor>>;
  dispose(): Promise<void>;
}

export async function createDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU unavailable: navigator.gpu undefined');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU unavailable: requestAdapter() returned null');
  return adapter.requestDevice();
}

export async function createSession(
  manifest: Manifest, image: number, baseUrl: string, device: GPUDevice,
): Promise<Runner> {
  const resolution = manifest.resolutions.find((r) => r.image === image);
  if (!resolution) throw new Error(`no resolution ${image} in manifest`);

  ort.env.webgpu.device = device;
  const session = await ort.InferenceSession.create(
    `${baseUrl}/${resolution.graph}`,
    {
      executionProviders: ['webgpu'],
      preferredOutputLocation: {
        grad_z: 'gpu-buffer', pred: 'gpu-buffer',
      },
    },
  );

  return {
    device, session, resolution,
    async run(z, target) {
      return session.run({ z, target });
    },
    async dispose() { await session.release(); },
  };
}
```

- [ ] **Step 4: Write the harness page**

`web/src/harness.ts`:

```ts
import * as ort from 'onnxruntime-web/webgpu';
import { loadManifest } from './manifest';
import { createDevice, createSession } from './session';

const BASE = '/models';
const TOL = 2e-5;

interface RefEntry { offset: number; count: number; shape: number[]; }

async function loadGolden(image: number) {
  const [meta, buf] = await Promise.all([
    fetch(`${BASE}/reference_${image}.json`).then((r) => r.json()),
    fetch(`${BASE}/reference_${image}.bin`).then((r) => r.arrayBuffer()),
  ]);
  const all = new Float32Array(buf);
  const get = (name: string) => {
    const e = meta[name] as RefEntry;
    if (!e) throw new Error(`golden tensor ${name} missing`);
    return { data: all.subarray(e.offset, e.offset + e.count), shape: e.shape };
  };
  return { get };
}

function maxAbs(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function main() {
  const cases: { name: string; maxAbs: number; pass: boolean; error?: string }[] = [];
  try {
    const manifest = await loadManifest(`${BASE}/manifest.json`);
    const image = manifest.resolutions[0].image;
    const golden = await loadGolden(image);
    const device = await createDevice();
    const runner = await createSession(manifest, image, BASE, device);

    const z = golden.get('z');
    const target = golden.get('target');
    const mk = (t: { data: Float32Array; shape: number[] }) =>
      new ort.Tensor('float32', t.data, t.shape);

    const out = await runner.run(mk(z), mk(target));
    for (const name of ['loss', 'score', 'grad_z', 'pred']) {
      const want = golden.get(name);
      const got = await out[name].getData(true) as Float32Array;
      const err = maxAbs(got, want.data);
      cases.push({ name, maxAbs: err, pass: err <= TOL });
    }

    // Negative control: a perturbed latent MUST fail.
    const bad = new Float32Array(z.data);
    bad[0] += 0.5;
    const outBad = await runner.run(
      new ort.Tensor('float32', bad, z.shape), mk(target));
    const gradBad = await outBad.grad_z.getData(true) as Float32Array;
    const errBad = maxAbs(gradBad, golden.get('grad_z').data);
    cases.push({
      name: 'negative control (perturbed z)',
      maxAbs: errBad, pass: errBad > TOL,
    });

    await runner.dispose();
  } catch (e) {
    cases.push({ name: 'harness', maxAbs: NaN, pass: false, error: String(e) });
  }

  const passed = cases.filter((c) => c.pass).length;
  (window as any).__RESULTS__ = {
    status: passed === cases.length ? 'ALL_PASS' : 'FAIL',
    passed, total: cases.length, cases,
  };
  document.title = `ONNX parity: ${(window as any).__RESULTS__.status}`;
  document.body.textContent = JSON.stringify((window as any).__RESULTS__, null, 2);
  (window as any).__DONE__ = true;
}

main();
```

`web/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
  <head><meta charset="utf-8"><title>ONNX parity</title></head>
  <body style="font:13px ui-monospace,monospace;white-space:pre-wrap;padding:16px">
    running…
    <script type="module" src="/src/harness.ts"></script>
  </body>
</html>
```

- [ ] **Step 5: Run the dev server and verify in Chrome**

```bash
cd D:/LODwebgpu/web && npm run dev -- --port 8732
```

Then via chrome-devtools MCP: `new_page` at `http://localhost:8732/`, then `evaluate_script`:

```js
async () => {
  const t0 = Date.now();
  while (!window.__DONE__ && Date.now() - t0 < 60000)
    await new Promise(r => setTimeout(r, 250));
  return window.__RESULTS__;
}
```

Expected: `status: "ALL_PASS"`, five cases — `loss`, `score`, `grad_z`, `pred` all under `2e-5`, and the negative control reporting a large error with `pass: true`.

**This is the gate.** If `loss`/`grad_z` do not match, or ORT-Web reports an unsupported op, stop and escalate before starting Milestone 1.

- [ ] **Step 6: Commit**

```bash
git add web/ docs/
git commit -m "feat: browser ONNX parity harness with negative control (Milestone 0 gate)"
```

---

# Milestone 1 — Headless Loop

### Task 9: Encoder export

**Files:**
- Modify: `export/to_onnx.py`
- Test: `smoke_test.py`

**Interfaces:**
- Produces: `export_encoder(out_dir: Path, model_id: str | None = None) -> Path` writing `encoder.onnx` with dynamic H/W; input `image` `[1,3,H,W]`, output `latent_mean` `[1,32,H/8,W/8]`

- [ ] **Step 1: Write the failing test**

```python
@test
def test_encoder_export_dynamic_shapes():
    """Encoder exports once and accepts both demo resolutions."""
    try:
        import onnxruntime as ort
        from diffusers.models.autoencoders.vae import Encoder
        import numpy as np
    except ImportError:
        print("      SKIP (onnxruntime/diffusers not installed)")
        return
    import sys, tempfile
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from export.to_onnx import export_encoder

    with tempfile.TemporaryDirectory() as td:
        path = export_encoder(Path(td))
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for size in (128, 256):
            img = np.random.randn(1, 3, size, size).astype(np.float32)
            (out,) = sess.run(None, {"image": img})
            assert out.shape == (1, 32, size // 8, size // 8), f"{size}: {out.shape}"
            print(f"      {size}x{size} -> {out.shape}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k encoder_export`
Expected: FAIL with `ImportError: cannot import name 'export_encoder'`

- [ ] **Step 3: Write minimal implementation**

Append to `export/to_onnx.py`:

```python
class _EncoderMean(torch.nn.Module):
    """Encoder -> posterior mean.  Deterministic: no sampling, so runs repeat."""

    def __init__(self, encoder, quant_conv=None, latent_channels: int = 32):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.latent_channels = latent_channels

    def forward(self, image):
        h = self.encoder(image)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        return h[:, : self.latent_channels]      # mean half of (mean, logvar)


def export_encoder(out_dir: Path, model_id: str | None = None) -> Path:
    from diffusers.models.autoencoders.vae import Encoder
    if model_id:
        from diffusers import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(
            model_id, torch_dtype=torch.float32).eval()
        module = _EncoderMean(vae.encoder, getattr(vae, "quant_conv", None),
                              vae.config.latent_channels)
    else:
        enc = Encoder(in_channels=3, out_channels=64,
                      down_block_types=("DownEncoderBlock2D",) * 4,
                      block_out_channels=(96, 192, 384, 384),
                      layers_per_block=2, norm_num_groups=32,
                      double_z=True).eval()
        module = _EncoderMean(enc, None, 32)
    for p in module.parameters():
        p.requires_grad_(False)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "encoder.onnx"
    torch.onnx.export(
        module, (torch.randn(1, 3, 128, 128),), str(path),
        input_names=["image"], output_names=["latent_mean"],
        dynamic_axes={"image": {2: "h", 3: "w"},
                      "latent_mean": {2: "lh", 3: "lw"}},
        opset_version=18, dynamo=False,
    )
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k encoder_export`
Expected: PASS, both `128x128 -> (1, 32, 16, 16)` and `256x256 -> (1, 32, 32, 32)`

- [ ] **Step 5: Commit**

```bash
git add export/to_onnx.py smoke_test.py
git commit -m "feat: export VAE encoder to ONNX with dynamic spatial axes"
```

---

### Task 10: Adam reference implementation

**Files:**
- Create: `export/adam_reference.py`
- Test: `smoke_test.py`

**Interfaces:**
- Produces: `adam_steps(z0, grads, lr, beta1, beta2, eps) -> np.ndarray` returning the latent after `len(grads)` steps, bias-corrected, matching `torch.optim.Adam`

- [ ] **Step 1: Write the failing test**

```python
@test
def test_adam_reference_matches_torch():
    """numpy Adam reproduces torch.optim.Adam over a fixed gradient sequence."""
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from export.adam_reference import adam_steps

    rng = np.random.default_rng(17)
    z0 = rng.standard_normal(64).astype(np.float32)
    grads = [rng.standard_normal(64).astype(np.float32) for _ in range(30)]
    cfg = dict(lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8)

    got = adam_steps(z0.copy(), grads, **cfg)

    zt = torch.tensor(z0.copy(), requires_grad=True)
    opt = torch.optim.Adam([zt], lr=cfg["lr"],
                           betas=(cfg["beta1"], cfg["beta2"]), eps=cfg["eps"])
    for g in grads:
        opt.zero_grad()
        zt.grad = torch.tensor(g)
        opt.step()
    want = zt.detach().numpy()

    err = float(np.abs(got - want).max())
    assert err < 1e-6, f"adam mismatch: {err:.3e}"
    print(f"      30 steps, max_abs_error={err:.2e}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k adam_reference`
Expected: FAIL with `ModuleNotFoundError: No module named 'export.adam_reference'`

- [ ] **Step 3: Write minimal implementation**

Create `export/adam_reference.py`:

```python
"""Reference Adam, shared by the Python tests and the WGSL kernel's golden data."""
from __future__ import annotations

import numpy as np


def adam_steps(z, grads, lr: float, beta1: float, beta2: float, eps: float):
    """Bias-corrected Adam, matching torch.optim.Adam defaults (no weight decay)."""
    z = np.asarray(z, dtype=np.float32).copy()
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    for t, g in enumerate(grads, start=1):
        g = np.asarray(g, dtype=np.float32)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1 ** t)
        vhat = v / (1.0 - beta2 ** t)
        z = z - lr * mhat / (np.sqrt(vhat) + eps)
    return z.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python smoke_test.py -k adam_reference`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add export/adam_reference.py smoke_test.py
git commit -m "feat: numpy Adam reference verified against torch.optim.Adam"
```

---

### Task 11: Adam WGSL kernel

**Files:**
- Create: `web/src/adam.wgsl`, `web/src/adam.ts`
- Modify: `web/src/harness.ts` (add the case)
- Test: browser, via chrome-devtools MCP

**Interfaces:**
- Consumes: `adam_steps` golden data from Task 10
- Produces: `class Adam { constructor(device, numel, cfg); step(zBuf, gradBuf): void; reset(): void }` — updates `zBuf` in place, owns `m`/`v`/step count

- [ ] **Step 1: Write the failing browser case**

Add to `web/src/harness.ts`, inside `main()` before the results are assembled:

```ts
    // --- Adam kernel vs numpy golden ---
    {
      const g = await fetch(`${BASE}/adam_golden.json`).then((r) => r.json());
      const { Adam } = await import('./adam');
      const n = g.z0.length;
      const zBuf = device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      device.queue.writeBuffer(zBuf, 0, new Float32Array(g.z0));
      const gradBuf = device.createBuffer({
        size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const adam = new Adam(device, n, g.cfg);
      for (const grad of g.grads) {
        device.queue.writeBuffer(gradBuf, 0, new Float32Array(grad));
        adam.step(zBuf, gradBuf);
      }
      const read = device.createBuffer({
        size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(zBuf, 0, read, 0, n * 4);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      const err = maxAbs(got, new Float32Array(g.expected));
      cases.push({ name: 'adam 30 steps', maxAbs: err, pass: err <= 1e-4 });

      const perturbed = new Float32Array(g.expected);
      perturbed[0] += 0.5;
      cases.push({
        name: 'negative control (adam)',
        maxAbs: maxAbs(got, perturbed),
        pass: maxAbs(got, perturbed) > 1e-4,
      });
    }
```

Generate the golden file (`stage_adam` was added to `export/make_fixtures.py` in Task 8):

```bash
cd D:/LODwebgpu
CONDA_NO_PLUGINS=true conda run -n pytorch-5090 --no-capture-output python export/make_fixtures.py adam
```

- [ ] **Step 2: Run to verify it fails**

Restart the dev server, reload the page via chrome-devtools MCP `navigate_page` with `type: "reload"`, then `evaluate_script` for `window.__RESULTS__`.
Expected: the harness case errors with `Failed to resolve module specifier './adam'`

- [ ] **Step 3: Write minimal implementation**

`web/src/adam.wgsl`:

```wgsl
struct Meta { n: u32, lr: f32, beta1: f32, beta2: f32, eps: f32,
              bc1: f32, bc2: f32, _pad: u32 };
@group(0) @binding(0) var<uniform> meta : Meta;
@group(0) @binding(1) var<storage, read_write> z : array<f32>;
@group(0) @binding(2) var<storage, read> grad : array<f32>;
@group(0) @binding(3) var<storage, read_write> m : array<f32>;
@group(0) @binding(4) var<storage, read_write> v : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= meta.n) { return; }
  let g = grad[i];
  let mi = meta.beta1 * m[i] + (1.0 - meta.beta1) * g;
  let vi = meta.beta2 * v[i] + (1.0 - meta.beta2) * g * g;
  m[i] = mi;
  v[i] = vi;
  let mhat = mi / meta.bc1;
  let vhat = vi / meta.bc2;
  z[i] = z[i] - meta.lr * mhat / (sqrt(vhat) + meta.eps);
}
```

`web/src/adam.ts`:

```ts
import wgsl from './adam.wgsl?raw';
import type { AdamConfig } from './manifest';

export class Adam {
  private pipeline: GPUComputePipeline;
  private meta: GPUBuffer;
  private m: GPUBuffer;
  private v: GPUBuffer;
  private t = 0;

  constructor(
    private device: GPUDevice,
    private numel: number,
    private cfg: AdamConfig,
  ) {
    const module = device.createShaderModule({ code: wgsl });
    this.pipeline = device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: 'main' },
    });
    this.meta = device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const zeros = new Float32Array(numel);
    const mk = () => {
      const b = device.createBuffer({
        size: numel * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(b, 0, zeros);
      return b;
    };
    this.m = mk();
    this.v = mk();
  }

  reset(): void {
    this.t = 0;
    const zeros = new Float32Array(this.numel);
    this.device.queue.writeBuffer(this.m, 0, zeros);
    this.device.queue.writeBuffer(this.v, 0, zeros);
  }

  step(zBuf: GPUBuffer, gradBuf: GPUBuffer): void {
    this.t += 1;
    const { lr, beta1, beta2, eps } = this.cfg;
    const buf = new ArrayBuffer(32);
    new Uint32Array(buf, 0, 1)[0] = this.numel;
    new Float32Array(buf, 4, 6).set([
      lr, beta1, beta2, eps,
      1 - Math.pow(beta1, this.t),
      1 - Math.pow(beta2, this.t),
    ]);
    this.device.queue.writeBuffer(this.meta, 0, buf);

    const bg = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.meta } },
        { binding: 1, resource: { buffer: zBuf } },
        { binding: 2, resource: { buffer: gradBuf } },
        { binding: 3, resource: { buffer: this.m } },
        { binding: 4, resource: { buffer: this.v } },
      ],
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(this.numel / 64));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
```

- [ ] **Step 4: Run to verify it passes**

Reload via chrome-devtools MCP and read `window.__RESULTS__`.
Expected: `adam 30 steps` under `1e-4`, `negative control (adam)` passing (large error).

- [ ] **Step 5: Commit**

```bash
git add web/src/adam.wgsl web/src/adam.ts web/src/harness.ts web/public/models/adam_golden.json
git commit -m "feat: Adam WGSL kernel verified against numpy reference in Chrome"
```

---

### Task 12: Latent initialization from the encoder

**Files:**
- Create: `web/src/latentInit.ts`
- Modify: `web/src/harness.ts`
- Test: browser

**Interfaces:**
- Consumes: `encoder.onnx` from Task 9, `createDevice` from Task 8
- Produces: `initLatent(device, encoderUrl, image: Float32Array, shape: number[], scaling: {factor: number; shift: number}) -> Promise<{buffer: GPUBuffer; numel: number}>`; the encoder session is **released before returning**

- [ ] **Step 1: Write the failing browser case**

Add to `web/src/harness.ts`:

```ts
    // --- encoder-seeded latent init ---
    {
      const { initLatent } = await import('./latentInit');
      const img = new Float32Array(1 * 3 * image * image).fill(0.25);
      const { buffer, numel } = await initLatent(
        device, `${BASE}/encoder.onnx`, img, [1, 3, image, image],
        { factor: 1.0, shift: 0.0 });
      const expected = 32 * (image / 8) * (image / 8);
      cases.push({
        name: 'latent init shape', maxAbs: Math.abs(numel - expected),
        pass: numel === expected,
      });
      const read = device.createBuffer({
        size: numel * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buffer, 0, read, 0, numel * 4);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const z0 = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      const finite = z0.every((x) => Number.isFinite(x));
      const nonzero = z0.some((x) => x !== 0);
      cases.push({
        name: 'latent init values finite and non-degenerate',
        maxAbs: finite && nonzero ? 0 : NaN, pass: finite && nonzero,
      });
    }
```

- [ ] **Step 2: Run to verify it fails**

Reload and read `window.__RESULTS__`.
Expected: failure resolving `./latentInit`

- [ ] **Step 3: Write minimal implementation**

`web/src/latentInit.ts`:

```ts
import * as ort from 'onnxruntime-web/webgpu';

export interface Scaling { factor: number; shift: number; }

/**
 * Encode an image to its posterior mean and upload it as the initial latent.
 *
 * Deterministic: takes the mean rather than sampling, so repeated runs on the
 * same image produce identical trajectories.  The encoder session is released
 * before returning — it is dead weight for the optimization loop, and freeing
 * it returns GPU memory exactly when the joint graph needs headroom.
 */
export async function initLatent(
  device: GPUDevice,
  encoderUrl: string,
  image: Float32Array,
  shape: number[],
  scaling: Scaling,
): Promise<{ buffer: GPUBuffer; numel: number }> {
  ort.env.webgpu.device = device;
  const session = await ort.InferenceSession.create(encoderUrl, {
    executionProviders: ['webgpu'],
  });
  let latent: Float32Array;
  try {
    const out = await session.run({
      image: new ort.Tensor('float32', image, shape),
    });
    const raw = (await out.latent_mean.getData(true)) as Float32Array;
    latent = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      latent[i] = (raw[i] - scaling.shift) * scaling.factor;
    }
  } finally {
    await session.release();
  }

  const buffer = device.createBuffer({
    size: latent.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, latent);
  return { buffer, numel: latent.length };
}
```

- [ ] **Step 4: Run to verify it passes**

Reload and read `window.__RESULTS__`.
Expected: both latent-init cases pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/latentInit.ts web/src/harness.ts
git commit -m "feat: encoder-seeded latent init, session released after use"
```

---

### Task 13: Headless 30-step loop and convergence

**Files:**
- Create: `web/src/loop.ts`
- Modify: `web/src/harness.ts`
- Test: browser

**Interfaces:**
- Consumes: `Runner` (Task 8), `Adam` (Task 11), `initLatent` (Task 12)
- Produces: `runLoop(runner, cfg: AdamConfig, z0: Float32Array, target: Float32Array, zShape: number[], targetShape: number[], device: GPUDevice) -> Promise<{losses: number[]; scores: number[]}>`

- [ ] **Step 1: Write the failing browser case**

Add to `web/src/harness.ts`:

```ts
    // --- 30-step headless optimization ---
    {
      const { runLoop } = await import('./loop');
      const { losses } = await runLoop(
        runner, manifest.adam, golden.get('z').data,
        golden.get('target').data, golden.get('z').shape,
        golden.get('target').shape, device);
      const improved = losses[losses.length - 1] < losses[0];
      const finite = losses.every((l) => Number.isFinite(l));
      cases.push({
        name: `loop ${losses.length} steps, loss ${losses[0].toFixed(5)} -> ${losses[losses.length - 1].toFixed(5)}`,
        maxAbs: losses[0] - losses[losses.length - 1],
        pass: finite && improved && losses.length === manifest.adam.steps,
      });
    }
```

- [ ] **Step 2: Run to verify it fails**

Reload; expected failure resolving `./loop`.

- [ ] **Step 3: Write minimal implementation**

Create `web/src/loop.ts`:

```ts
import * as ort from 'onnxruntime-web/webgpu';
import { Adam } from './adam';
import type { AdamConfig } from './manifest';
import type { Runner } from './session';

export async function runLoop(
  runner: Runner,
  cfg: AdamConfig,
  z0: Float32Array,
  target: Float32Array,
  zShape: number[],
  targetShape: number[],
  device: GPUDevice,
): Promise<{ losses: number[]; scores: number[] }> {
  const numel = z0.length;
  const zBuf = device.createBuffer({
    size: numel * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(zBuf, 0, z0);

  const adam = new Adam(device, numel, cfg);
  const targetTensor = new ort.Tensor('float32', target, targetShape);
  const losses: number[] = [];
  const scores: number[] = [];

  // z is read back each step only because the session takes a CPU tensor here;
  // Milestone 2 replaces this with a gpu-buffer input binding.
  let z = z0.slice();
  for (let step = 0; step < cfg.steps; step++) {
    const out = await runner.run(new ort.Tensor('float32', z, zShape), targetTensor);
    losses.push(((await out.loss.getData(true)) as Float32Array)[0]);
    scores.push(((await out.score.getData(true)) as Float32Array)[0]);

    const grad = (await out.grad_z.getData(true)) as Float32Array;
    const gradBuf = device.createBuffer({
      size: numel * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(gradBuf, 0, grad);
    adam.step(zBuf, gradBuf);

    const read = device.createBuffer({
      size: numel * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(zBuf, 0, read, 0, numel * 4);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    z = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();
    gradBuf.destroy();

    if (!Number.isFinite(losses[losses.length - 1])) {
      throw new Error(`loss diverged to NaN at step ${step}`);
    }
  }
  return { losses, scores };
}
```

- [ ] **Step 4: Run to verify it passes**

Reload and read `window.__RESULTS__`.
Expected: the loop case passes with a decreasing loss across 30 steps.

If loss does not decrease, the Adam fixture `lr` is wrong for this graph — that is the expected symptom of the unresolved hyperparameter question, not a code bug. Record the observed behaviour and escalate for the real values.

- [ ] **Step 5: Record the convergence numbers**

```bash
cd D:/LODwebgpu
echo "Milestone 1 convergence (fixture hyperparameters, NOT reference LOD values):" > docs/superpowers/m1-convergence.md
```

Paste the `losses` and `scores` arrays from `window.__RESULTS__` into that file with the adapter string and date.

- [ ] **Step 6: Commit**

```bash
git add web/src/loop.ts web/src/harness.ts docs/superpowers/m1-convergence.md
git commit -m "feat: headless 30-step optimization loop with convergence record"
```

---

## Self-Review

**Spec coverage.** §1 architecture → Tasks 4, 8. §1 Python change → Tasks 1, 2. §2 components: `export/wrap.py` T4, `export/to_onnx.py` T5–6, T9, `export/reference.py` T7, `session.ts` T8, `adam.*` T11, `latentInit.ts` T12, `manifest.ts` T8. §3 payload → T6 (fp16 + external data). §4 pipeline → T12, T13. §5 static shapes → T6. §7 testing: ONNX parity T5, fp16 round trip T6, Adam reference T10, shared external data T6, end-to-end step parity T8, negative controls T8/T11. §8 Milestone 0 steps 1–4 → T2, T5, T8, T6.

**Deferred to the Milestone 2–4 plan, by design:** `display.wgsl`/`display.ts`, `telemetry.ts`, `imageInput.ts`, `app.ts`, all of §6 error handling, the display-shader browser case, resolution auto-selection, and the manual integrated-GPU gate. Task 13 reads `z` back to the CPU each step as a documented interim; the gpu-buffer input binding that removes it belongs with the display work.

**Known gaps carried forward:** Adam hyperparameters remain fixtures (Global Constraints, T13 step 4). The level-3 score *reducer* is a defensible definition but unconfirmed against reference LOD (T1 note).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-17-browser-lod-runtime-m0-m1.md`.
