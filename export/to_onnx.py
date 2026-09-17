"""Export the rewritten joint graph to ONNX."""
from __future__ import annotations

import importlib.util
import json
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

from export.wrap import ExportWrapper  # noqa: E402

OUTPUT_NAMES = ["loss", "pred", "score", "grad_z"]
INPUT_NAMES = ["z", "target"]


def _placeholder_args(gm, sig, module, z, target, mode):
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
        args = _placeholder_args(gm, sig, m, fz, ft, mode)
    # meta['val'] is required by the dynamo/torch.export ONNX path.
    rw.rewrite(gm, example_args=args)
    wrapper = ExportWrapper(gm, sig, m).eval()
    return gm, wrapper, z, t


def export_joint(res: int, out_dir: Path, fp16: bool = False,
                 external_data: bool = False, seed: int = 0) -> Path:
    gm, wrapper, z, t = build_graph(res, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lod_joint_{res * 8}.onnx"
    # The legacy TorchScript exporter has no symbolic for aten::native_group_norm;
    # the dynamo path maps it to GroupNormalization.  Try dynamo first, keep the
    # legacy path as a fallback so the failure mode stays visible.
    try:
        torch.onnx.export(
            wrapper, (z, t), str(path),
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            opset_version=18, dynamo=True,
        )
    except Exception as dynamo_exc:                      # pragma: no cover
        try:
            torch.onnx.export(
                wrapper, (z, t), str(path),
                input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                opset_version=18, dynamo=False, do_constant_folding=False,
            )
        except Exception as legacy_exc:
            raise RuntimeError(
                "both ONNX export paths failed. "
                f"dynamo=True: {type(dynamo_exc).__name__}: {dynamo_exc} | "
                f"dynamo=False: {type(legacy_exc).__name__}: {legacy_exc}"
            ) from dynamo_exc
    if fp16 or external_data:
        finalize(path, fp16=fp16, external_data=external_data)
    return path


def _to_fp16_initializers(model) -> None:
    """Store initializers as fp16 and insert Cast->fp32 so compute stays fp32."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    graph = model.graph
    renamed = []
    for init in graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        original = init.name
        half = numpy_helper.from_array(arr, original + "_fp16")
        init.CopyFrom(half)
        renamed.append(original)

    for original in renamed:
        out = original + "_fp32"
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == original:
                    node.input[i] = out
        import onnx as _onnx
        graph.node.insert(0, _onnx.helper.make_node(
            "Cast", [original + "_fp16"], [out], to=_onnx.TensorProto.FLOAT,
            name="cast_" + original))


def finalize(path: Path, fp16: bool, external_data: bool) -> None:
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    model = onnx.load(str(path))
    if fp16:
        _to_fp16_initializers(model)
    if external_data:
        convert_model_to_external_data(
            model, all_tensors_to_one_file=True, location="weights.bin",
            size_threshold=1024, convert_attribute=False)
    onnx.save(model, str(path))


def write_manifest(out_dir: Path, resolutions: list, adam: dict) -> Path:
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
