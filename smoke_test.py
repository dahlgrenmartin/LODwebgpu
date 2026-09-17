#!/usr/bin/env python3
"""Smoke test for the FLUX.2-small LOD -> WebGPU rewrite pipeline.

Unlike the original validate_flux2_rewrites.py (which only printed numbers),
every check here asserts and the process exits non-zero on failure.

    python smoke_test.py            # all tests
    python smoke_test.py -k sym4    # substring filter
    python smoke_test.py --json out.json
"""
from __future__ import annotations

import argparse
import collections
import copy
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode

ROOT = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load("probe", "flux2_lod_aot_probe.py")
rw = _load("rw", "flux2_targeted_rewrites.py")

TESTS = []
RESULTS = {}


def test(fn):
    TESTS.append(fn)
    return fn


def _counts(gm):
    return collections.Counter(rw.opname(n) for n in gm.graph.nodes if rw.opname(n))


def _backward_ops(gm):
    return {k: v for k, v in _counts(gm).items() if "backward" in k}


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
        args.append(mode.from_tensor(t) if mode is not None else t)
    return args


def _capture(module, z, target):
    with FakeTensorMode(allow_non_fake_inputs=True):
        gm, sig = aot_export_module(module, (z, target), trace_joint=True, output_loss_index=0)
    return gm, sig


@test
def test_sym4_matches_pywt():
    """Sym4 detail filters reproduce pywt.dwt2(mode='zero') coefficient-exactly."""
    try:
        import pywt
        import numpy as np
    except ImportError:
        print("      SKIP (pywt not installed)")
        RESULTS["sym4"] = "skipped"
        return
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 32, 32))
    kernels = probe.Sym4Level1Loss().kernels.to(torch.float64)
    xt = torch.tensor(x, dtype=torch.float64)
    d = F.conv2d(F.pad(xt, (6, 6, 6, 6)), kernels, stride=2, groups=3)

    _, (cH, cV, cD) = pywt.dwt2(x[0, 0], "sym4", mode="zero")
    got = [d[0, i].numpy() for i in range(3)]
    errs_out = {}
    for name, ref in (("cH", cH), ("cV", cV), ("cD", cD)):
        errs = [np.abs(g - ref).max() if g.shape == ref.shape else np.inf for g in got]
        best = int(np.argmin(errs))
        assert errs[best] < 1e-6, f"{name}: no matching detail band (best err {errs[best]:.3e})"
        errs_out[name] = float(errs[best])
        print(f"      {name}: kernel {best}, shape={ref.shape}, max_err={errs[best]:.2e}")
    RESULTS["sym4"] = errs_out


@test
def test_structural_rewrite_eliminates_backward():
    """Structural probe graph lowers with zero *_backward ops remaining."""
    m = probe.JointLOD("sym4").eval()
    with FakeTensorMode(allow_non_fake_inputs=True):
        z = torch.empty(1, 32, 8, 8, requires_grad=True)
        t = torch.empty(1, 3, 64, 64)
        gm, _ = aot_export_module(m, (z, t), trace_joint=True, output_loss_index=0)
    pre = _backward_ops(gm)
    assert pre, "expected backward ops before rewrite"
    stats = rw.rewrite(gm)
    left = _backward_ops(gm)
    assert not left, f"backward ops survived: {left}"
    for key, want in (("conv_bwd", 37), ("gn_bwd", 30), ("softmax_bwd", 1), ("sigmoid_bwd", 29)):
        assert stats[key] == want, f"{key}={stats[key]}, expected {want}"
    scalar = {k: v for k, v in stats.items() if k != "layout"}
    print(f"      lowered {dict(pre)} -> {{}}")
    print(f"      stats={scalar}")
    RESULTS["structural"] = scalar


@test
def test_rewrite_is_numerically_exact():
    """Rewritten graph matches the original joint graph to fp32 roundoff."""
    torch.manual_seed(123)
    m = probe.JointLOD("sym4").eval()
    z = torch.randn(1, 32, 2, 2, requires_grad=True)
    t = torch.randn(1, 3, 16, 16)
    gm, sig = aot_export_module(m, (z, t), trace_joint=True, output_loss_index=0)
    gm2 = copy.deepcopy(gm)
    rw.rewrite(gm2)

    state = m.state_dict()
    umap = {sig.user_inputs[0]: z.detach(), sig.user_inputs[1]: t}
    args = []
    for n in gm.graph.nodes:
        if n.op != "placeholder":
            continue
        if n.name in sig.inputs_to_parameters:
            args.append(state[sig.inputs_to_parameters[n.name]])
        elif n.name in sig.inputs_to_buffers:
            args.append(state[sig.inputs_to_buffers[n.name]])
        elif n.name in umap:
            args.append(umap[n.name])
        else:
            raise KeyError(n.name)

    with torch.no_grad():
        ref, got = gm(*args), gm2(*args)
    assert len(ref) == len(got), f"output count changed: {len(ref)} -> {len(got)}"

    # loss, pred, score are forward values (bit-identical); grad_z is the
    # rewritten backward, allowed fp32 roundoff.
    tol = [0.0, 0.0, 0.0, 1e-7]
    errs = []
    for i, (a, b) in enumerate(zip(ref, got)):
        assert isinstance(a, torch.Tensor) == isinstance(b, torch.Tensor), f"output[{i}] type changed"
        if not isinstance(a, torch.Tensor):
            assert a == b, f"output[{i}] non-tensor mismatch {a} != {b}"
            continue
        err = (a - b).abs().max().item()
        assert err <= tol[i], f"output[{i}] max_abs_error={err:.3e} > tol {tol[i]:.3e}"
        errs.append(err)
        print(f"      output[{i}] shape={tuple(a.shape)} max_abs_error={err:.3e} (tol {tol[i]:.1e})")
    RESULTS["numerical"] = errs


@test
def test_groupnorm_rank3_lowers():
    """Rank-3 GroupNorm (the Diffusers AttnProcessor path) lowers, not just NCHW."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gn = torch.nn.GroupNorm(4, 16, eps=1e-6)
            for p in self.gn.parameters():
                p.requires_grad_(False)

        def forward(self, z, t):
            y = self.gn(z)
            return (y - t).abs().mean(), y.mean().detach()

    with FakeTensorMode(allow_non_fake_inputs=True):
        z = torch.empty(1, 16, 9, requires_grad=True)
        t = torch.empty(1, 16, 9)
        gm, _ = aot_export_module(M().eval(), (z, t), trace_joint=True, output_loss_index=0)
    assert any("group_norm_backward" in k for k in _counts(gm)), "no groupnorm backward captured"
    rw.rewrite(gm)
    assert not _backward_ops(gm), f"backward survived: {_backward_ops(gm)}"
    print("      rank-3 [1,16,9] GroupNorm backward lowered")


@test
def test_groupnorm_non_affine_lowers():
    """affine=False GroupNorm (weight=None) lowers instead of crashing."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gn = torch.nn.GroupNorm(4, 16, eps=1e-6, affine=False)

        def forward(self, z, t):
            y = self.gn(z)
            return (y - t).abs().mean(), y.mean().detach()

    with FakeTensorMode(allow_non_fake_inputs=True):
        z = torch.empty(1, 16, 4, 4, requires_grad=True)
        t = torch.empty(1, 16, 4, 4)
        gm, _ = aot_export_module(M().eval(), (z, t), trace_joint=True, output_loss_index=0)
    rw.rewrite(gm)
    assert not _backward_ops(gm), f"backward survived: {_backward_ops(gm)}"
    print("      affine=False GroupNorm backward lowered")


@test
def test_real_diffusers_decoder_lowers():
    """The real Diffusers Decoder (the class AutoencoderKLFlux2 uses) fully lowers."""
    try:
        from diffusers.models.autoencoders.vae import Decoder
        from diffusers.models.attention_processor import AttnProcessor
    except ImportError:
        print("      SKIP (diffusers not installed)")
        RESULTS["real_decoder"] = "skipped"
        return

    d = Decoder(
        in_channels=8, out_channels=3, up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(32, 32, 64), layers_per_block=1, norm_num_groups=32,
    ).eval()
    for p in d.parameters():
        p.requires_grad_(False)
    for mod in d.modules():
        if hasattr(mod, "set_processor"):
            mod.set_processor(AttnProcessor())

    class J(torch.nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.d = dec

        def forward(self, z, t):
            pred = self.d(z)
            return (pred - t).abs().mean(), pred.mean().detach()

    with FakeTensorMode(allow_non_fake_inputs=True):
        z = torch.empty(1, 8, 4, 4, requires_grad=True)
        t = torch.empty(1, 3, 16, 16)
        gm, _ = aot_export_module(J(d).eval(), (z, t), trace_joint=True, output_loss_index=0)
    pre = _backward_ops(gm)
    assert pre, "expected backward ops in the real decoder graph"
    rw.rewrite(gm)
    left = _backward_ops(gm)
    assert not left, f"backward ops survived on the real decoder: {left}"
    print(f"      real Decoder: {dict(pre)} -> {{}}")
    RESULTS["real_decoder"] = {k: v for k, v in pre.items()}


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


@test
def test_joint_outputs_image_and_score():
    """Joint graph exposes loss, pred image, detector score, and grad_z."""
    m = probe.JointLOD("sym4").eval()
    with FakeTensorMode(allow_non_fake_inputs=True):
        z = torch.empty(1, 32, 8, 8, requires_grad=True)
        t = torch.empty(1, 3, 64, 64)
        gm, _ = aot_export_module(m, (z, t), trace_joint=True, output_loss_index=0)
    out_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    flat = list(out_node.args[0])
    assert len(flat) == 4, f"expected 4 graph outputs, got {len(flat)}"
    shapes = [tuple(n.meta["val"].shape) for n in flat]
    assert shapes[1] == (1, 3, 64, 64), f"pred shape {shapes[1]}"
    assert shapes[0] == () and shapes[2] == (), f"loss/score not scalar: {shapes}"
    assert shapes[3] == (1, 32, 8, 8), f"grad_z shape {shapes[3]}"
    rw.rewrite(gm)
    assert not _backward_ops(gm), f"backward survived: {_backward_ops(gm)}"
    print(f"      outputs: loss{shapes[0]} pred{shapes[1]} score{shapes[2]} grad_z{shapes[3]}")


@test
def test_rewrite_repopulates_meta():
    """Every rewritten node carries meta['val'] so export can consume the graph."""
    m = probe.JointLOD("sym4").eval()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        z = torch.empty(1, 32, 4, 4, requires_grad=True)
        t = torch.empty(1, 3, 32, 32)
        gm, sig = aot_export_module(m, (z, t), trace_joint=True, output_loss_index=0)
        args = _placeholder_args(gm, sig, m, z, t, mode)
    rw.rewrite(gm, example_args=args)
    missing = [n.name for n in gm.graph.nodes
               if n.op == "call_function" and "val" not in n.meta]
    assert not missing, f"{len(missing)} nodes without meta['val']: {missing[:5]}"
    total = sum(1 for n in gm.graph.nodes if n.op == "call_function")
    print(f"      all {total} call_function nodes have meta['val']")


@test
def test_export_wrapper_matches_graph():
    """Wrapper with 2 inputs reproduces the lifted graph's outputs exactly."""
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
    print("      4 outputs identical; wrapper exposes 2 inputs")


@test
def test_onnx_matches_fx_graph():
    """Exported ONNX reproduces the FX graph under onnxruntime CPU EP."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("      SKIP (onnxruntime not installed)")
        return
    import tempfile
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

    for i, name in enumerate(["loss", "pred", "score", "grad_z"]):
        a = ref[i].numpy()
        b = np.asarray(got[i])
        denom = max(1e-6, float(np.abs(a).max()))
        rel = float(np.abs(a - b).max()) / denom
        assert rel < 1e-4, f"{name}: rel err {rel:.3e}"
        print(f"      {name}: rel_err={rel:.2e} shape={b.shape}")


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
    import tempfile
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

        def dtypes(path):
            m = onnx.load(str(path), load_external_data=False)
            return {i.name: i.data_type for i in m.graph.initializer}
        a, b = dtypes(p32), dtypes(p16)
        shared = set(a) & set(b)
        assert shared, "no initializer names shared between resolutions"
        for n in shared:
            assert a[n] == b[n], f"{n}: dtype differs between resolutions"
        half = sum(1 for v in a.values() if v == onnx.TensorProto.FLOAT16)
        assert half > 0, "no fp16 initializers produced"
        print(f"      {len(shared)}/{len(a)} initializers shared, {half} stored fp16")
        print(f"      graph files {[s // 1024 for s in sizes]} KiB, weights.bin "
              f"{(d / 'weights.bin').stat().st_size // 1024} KiB")

        _, wrapper, z, t = build_graph(res=32, seed=5)
        with torch.no_grad():
            ref = wrapper(z, t)
        sess = ort.InferenceSession(str(p32), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"z": z.numpy(), "target": t.numpy()})
        r, g = ref[3].numpy(), np.asarray(got[3])
        rel = float(np.abs(r - g).max()) / max(1e-6, float(np.abs(r).max()))
        # Measured 2026-09-17: fp16 weight storage costs 2.6e-03 at 128x128 and
        # 4.2e-02 at 256x256 on grad_z, while loss/pred/score stay <= 1.6e-03.
        # Accepted trade for halving the download; see the design doc.  The
        # threshold guards against regression, not against the known cost.
        assert rel < 6e-2, f"fp16 initializers degraded grad_z: rel {rel:.3e}"
        print(f"      fp16 grad_z rel_err={rel:.2e} (accepted cost, budget 6e-2)")


@test
def test_reference_dump_roundtrips():
    """Golden binary matches the wrapper outputs it was produced from."""
    import tempfile
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
        print(f"      {len(named)} tensors, {raw.size} floats total")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="only run tests whose name contains this")
    ap.add_argument("--json", default=None, help="write machine-readable results here")
    args = ap.parse_args()

    selected = [t for t in TESTS if args.k in t.__name__]
    print(f"torch={torch.__version__}  running {len(selected)}/{len(TESTS)} tests\n")

    failed = []
    for t in selected:
        print(f"[ RUN  ] {t.__name__}")
        print(f"         {(t.__doc__ or '').strip()}")
        try:
            t()
            print("[  OK  ]\n")
        except Exception as exc:
            failed.append(t.__name__)
            print(f"[ FAIL ] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print()

    if args.json:
        Path(args.json).write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")

    if failed:
        print(f"FAILED {len(failed)}/{len(selected)}: {', '.join(failed)}")
        return 1
    print(f"PASSED {len(selected)}/{len(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
