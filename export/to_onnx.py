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

# Initializers below this size stay fp32: they are scalars and shape
# constants, contribute nothing to download size, and can overflow fp16.
FP16_MIN_NUMEL = 1024
FP16_MAX = 65504.0


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


def patch_upsamplers_for_dynamic(module) -> int:
    """Express Diffusers' nearest x2 upsample as view -> expand -> reshape.

    F.interpolate(mode="nearest") decomposes to aten._unsafe_index with computed
    index tensors, and those cannot broadcast under symbolic shapes -- dynamic
    capture dies with "SymIntArrayRef expected to contain only concrete
    integers".  The view/expand/reshape formulation is bit-identical for an
    integer x2 nearest upsample and is shape-generic, which is exactly the trick
    the structural probe already uses.
    """
    from diffusers.models.upsampling import Upsample2D

    def forward(self, hidden_states, output_size=None, *args, **kwargs):
        assert hidden_states.shape[1] == self.channels
        if self.norm is not None:
            hidden_states = self.norm(
                hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.use_conv_transpose:
            return self.conv(hidden_states)
        if self.interpolate:
            if output_size is not None:
                raise NotImplementedError(
                    'explicit output_size is not supported on the dynamic path')
            n, c, h, w = hidden_states.shape
            hidden_states = (hidden_states
                             .view(n, c, h, 1, w, 1)
                             .expand(n, c, h, 2, w, 2)
                             .reshape(n, c, h * 2, w * 2))
        if self.use_conv:
            hidden_states = (self.conv(hidden_states) if self.name == 'conv'
                             else self.Conv2d_0(hidden_states))
        return hidden_states

    patched = 0
    for m in module.modules():
        if isinstance(m, Upsample2D):
            m.forward = forward.__get__(m, type(m))
            patched += 1
    return patched


def build_graph_dynamic(res: int = 16, seed: int = 0, real: bool = True,
                        model_id: str = "black-forest-labs/FLUX.2-small-decoder"):
    """Capture the joint graph with symbolic H/W so one ONNX serves any size.

    The rewrite is shape-static only in the sense that it reads meta['val'];
    when those are SymInts it still lowers, because every decoder convolution is
    stride 1 and its output_padding collapses to 0 at any resolution.
    """
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    torch.manual_seed(seed)
    if real:
        vae = load_real_vae(model_id)
        m = RealJointLOD(vae).eval()
        n_patched = patch_upsamplers_for_dynamic(m)
        if not n_patched:
            raise RuntimeError('no Upsample2D found to patch for dynamic capture')
        latent_c, out_c = vae.config.latent_channels, vae.config.out_channels
    else:
        m = probe.JointLOD("sym4").eval()
        latent_c, out_c = 32, 3

    # Real tensors created OUTSIDE the fake mode: from_tensor only assigns fresh
    # symbols when it is handed a concrete tensor.
    rz = torch.empty(1, latent_c, res, res, requires_grad=True)
    rt = torch.empty(1, out_c, res * 8, res * 8)
    mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
    with mode:
        fz = mode.from_tensor(rz, static_shapes=False)
        # Derive the target's spatial dims from the latent's rather than giving
        # them independent symbols.  The decoder upsamples by exactly 8, and
        # encoding that relation lets the shape env prove H is a multiple of 8 --
        # which is what makes the stride-2 wavelet convolution's output_padding a
        # constant instead of a parity-dependent expression.
        ft = torch.empty(1, out_c, fz.shape[2] * 8, fz.shape[3] * 8)
        gm, sig = aot_export_module(m, (fz, ft), trace_joint=True, output_loss_index=0)
    rw.rewrite(gm)
    wrapper = ExportWrapper(gm, sig, m).eval()
    z = torch.randn(1, latent_c, res, res)
    t = torch.randn(1, out_c, res * 8, res * 8)
    return gm, wrapper, z, t


def export_joint_dynamic(out_dir: Path, res: int = 16, fp16: bool = False,
                         external_data: bool = False, seed: int = 0,
                         max_latent: int = 128, real: bool = True) -> Path:
    """One graph for every image size up to max_latent*8 on each side."""
    from torch.export import Dim

    gm, wrapper, z, t = build_graph_dynamic(res=res, seed=seed, real=real)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lod_joint_dynamic.onnx"

    zh = Dim("zh", min=2, max=max_latent)
    zw = Dim("zw", min=2, max=max_latent)
    torch.onnx.export(
        wrapper, (z, t), str(path),
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        dynamic_shapes={"z": {2: zh, 3: zw},
                        "target": {2: 8 * zh, 3: 8 * zw}},
        opset_version=18, dynamo=True,
    )
    if fp16 or external_data:
        finalize(path, fp16=fp16, external_data=external_data)
    return path


def RealJointLOD_placeholder():  # pragma: no cover
    pass


class RealJointLOD(torch.nn.Module):
    """The real FLUX.2-small VAE decoder wired to the LOD loss and detector.

    Mirrors AutoencoderKLFlux2._decode exactly: post_quant_conv then decoder.
    Everything is frozen, so every backward node is input-gradient-only and the
    rewrite applies unchanged.
    """

    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = getattr(vae, "post_quant_conv", None)
        self.decoder = vae.decoder
        self.sym4 = probe.Sym4Level1Loss()
        self.detector = probe.Sym4Level3Detector(channels=3)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z, target):
        h = z if self.post_quant_conv is None else self.post_quant_conv(z)
        pred = self.decoder(h)
        loss = self.sym4(pred - target)
        score = self.detector(pred)
        return loss, pred.detach(), score.detach()


def load_real_vae(model_id: str = "black-forest-labs/FLUX.2-small-decoder"):
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    # Force the explicit bmm/softmax attention path; the fused SDPA kernel emits
    # _scaled_dot_product_flash_attention_backward, which nothing lowers.
    try:
        vae.set_default_attn_processor()
    except Exception:
        pass
    return vae


def build_graph_real(res: int, seed: int = 0,
                     model_id: str = "black-forest-labs/FLUX.2-small-decoder"):
    """Same contract as build_graph, but with the trained checkpoint."""
    torch.manual_seed(seed)
    vae = load_real_vae(model_id)
    m = RealJointLOD(vae).eval()
    z = torch.randn(1, vae.config.latent_channels, res, res)
    t = torch.randn(1, vae.config.out_channels, res * 8, res * 8)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        fz = torch.empty(1, vae.config.latent_channels, res, res, requires_grad=True)
        ft = torch.empty(1, vae.config.out_channels, res * 8, res * 8)
        gm, sig = aot_export_module(m, (fz, ft), trace_joint=True, output_loss_index=0)
    rw.rewrite(gm)
    wrapper = ExportWrapper(gm, sig, m).eval()
    return gm, wrapper, z, t


def export_joint_real(res: int, out_dir: Path, fp16: bool = False,
                      external_data: bool = False, seed: int = 0,
                      model_id: str = "black-forest-labs/FLUX.2-small-decoder") -> Path:
    gm, wrapper, z, t = build_graph_real(res, seed=seed, model_id=model_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lod_joint_{res * 8}.onnx"
    torch.onnx.export(
        wrapper, (z, t), str(path),
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        opset_version=18, dynamo=True,
    )
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
        arr = numpy_helper.to_array(init)
        # Only bulk tensors are worth halving.  Scalar/small constants carry all
        # the overflow risk and none of the download cost: the Sym4 loss reduces
        # over ~154k elements, and that divisor exceeds fp16's 65504 max, which
        # silently becomes inf and NaNs the whole graph.
        if arr.size < FP16_MIN_NUMEL:
            continue
        if not np.isfinite(arr).all() or float(np.abs(arr).max()) > FP16_MAX:
            continue
        original = init.name
        half = numpy_helper.from_array(arr.astype(np.float16), original + "_fp16")
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


def finalize(path: Path, fp16: bool, external_data: bool,
             location: str = "weights.bin") -> None:
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    model = onnx.load(str(path))          # pulls in any exporter-written sidecar
    if fp16:
        _to_fp16_initializers(model)
    if external_data:
        # onnx.save appends to an existing external-data file rather than
        # truncating it, so re-running an export silently doubles the payload.
        stale_weights = Path(path).parent / location
        if stale_weights.exists():
            stale_weights.unlink()
        convert_model_to_external_data(
            model, all_tensors_to_one_file=True, location=location,
            size_threshold=1024, convert_attribute=False)
    onnx.save(model, str(path))

    # The dynamo exporter writes its own fp32 sidecar (<model>.onnx.data).  Once
    # the weights live in our shared weights.bin nothing references it, and it
    # would otherwise sit in the served directory as ~112 MB of dead payload.
    if external_data:
        stale = Path(str(path) + ".data")
        if stale.exists():
            stale.unlink()


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


def export_encoder(out_dir: Path, model_id: str = None, fp16: bool = False,
                   external_data: bool = False) -> Path:
    from diffusers.models.autoencoders.vae import Encoder
    if model_id:
        vae = load_real_vae(model_id)
        module = _EncoderMean(vae.encoder, getattr(vae, "quant_conv", None),
                              vae.config.latent_channels)
    else:
        # double_z=True makes conv_out emit 2*out_channels = (mean, logvar).
        enc = Encoder(in_channels=3, out_channels=32,
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
        opset_version=18, dynamo=True,
    )
    if fp16 or external_data:
        finalize(path, fp16=fp16, external_data=external_data,
                 location="encoder_weights.bin")
    return path


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
        "encoderWeights": "encoder_weights.bin",
        "adam": adam,
        "outputs": OUTPUT_NAMES,
        "inputs": INPUT_NAMES,
    }
    p = Path(out_dir) / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p
