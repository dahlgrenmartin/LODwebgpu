"""Static width x height ONNX export helpers for the ORT-Web backend.

The WGSL backend remains shape-agnostic. ORT-Web gets one static graph per
supported image shape because the fixed graphs use substantially less graph
planning machinery than a fully dynamic ONNX model.
"""
from __future__ import annotations

import json
from pathlib import Path

INPUT_NAMES = ["z", "target"]
OUTPUT_NAMES = ["loss", "pred", "score", "grad_z"]


def _validate_size(width: int, height: int) -> tuple[int, int]:
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if width % 8 or height % 8:
        raise ValueError(f"{width}x{height} must have both sides as a multiple of 8")
    return width // 8, height // 8


def graph_filename(width: int, height: int) -> str:
    _validate_size(width, height)
    return f"lod_joint_{width}x{height}.onnx"


def write_manifest(out_dir: Path, sizes: list[tuple[int, int]], adam: dict,
                   latent_channels: int = 32,
                   weights: list[str] | str = "weights.bin") -> Path:
    """Write the browser manifest for exact ORT image sizes.

    ``sizes`` are image ``(width, height)`` pairs. Latent dimensions are NCHW,
    so the stored latent shape is ``[1, latent_channels, height/8, width/8]``.
    The channel count is model-specific: FLUX.2-small uses 32, SD 1.5 and
    SDXL use 4.
    """
    for key in ("lr", "beta1", "beta2", "eps", "steps"):
        if key not in adam:
            raise KeyError(f"adam.{key} is required and has no default")

    resolutions = []
    seen: set[tuple[int, int]] = set()
    for width, height in sizes:
        latent_w, latent_h = _validate_size(width, height)
        key = (width, height)
        if key in seen:
            raise ValueError(f"duplicate ORT size {width}x{height}")
        seen.add(key)
        resolutions.append({
            "width": width,
            "height": height,
            "latent": [1, latent_channels, latent_h, latent_w],
            "numel": latent_channels * latent_h * latent_w,
            "graph": graph_filename(width, height),
        })

    manifest = {
        "resolutions": resolutions,
        # A list: the shared blob is sharded when it would exceed GitHub's
        # 100 MB per-file limit, which SDXL's decoder does on its own.
        "weights": [weights] if isinstance(weights, str) else list(weights),
        "encoder": "encoder.onnx",
        "encoderWeights": "encoder_weights.bin",
        "adam": adam,
        "outputs": OUTPUT_NAMES,
        "inputs": INPUT_NAMES,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_graph_real_shape(
    width: int,
    height: int,
    *,
    seed: int = 0,
    model_id: str = "black-forest-labs/FLUX.2-small-decoder",
    vae=None,
):
    """Capture one real FLUX.2 joint graph for an exact image shape."""
    latent_w, latent_h = _validate_size(width, height)

    import torch
    from torch._functorch.aot_autograd import aot_export_module
    from torch._subclasses.fake_tensor import FakeTensorMode
    from export import to_onnx as base

    torch.manual_seed(seed)
    if vae is None:
        vae = base.load_real_vae(model_id)
    module = base.RealJointLOD(vae).eval()
    latent_c = vae.config.latent_channels
    out_c = vae.config.out_channels

    z = torch.randn(1, latent_c, latent_h, latent_w)
    target = torch.randn(1, out_c, height, width)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        fz = torch.empty(1, latent_c, latent_h, latent_w, requires_grad=True)
        ft = torch.empty(1, out_c, height, width)
        gm, sig = aot_export_module(
            module, (fz, ft), trace_joint=True, output_loss_index=0)

    base.rw.rewrite(gm)
    wrapper = base.ExportWrapper(gm, sig, module).eval()
    return gm, wrapper, z, target


def export_joint_real_shape(
    width: int,
    height: int,
    out_dir: Path,
    *,
    fp16: bool = False,
    external_data: bool = False,
    seed: int = 0,
    model_id: str = "black-forest-labs/FLUX.2-small-decoder",
    vae=None,
) -> Path:
    """Export one static real-checkpoint graph for ``width x height`` pixels."""
    import torch
    from export import to_onnx as base

    _, wrapper, z, target = build_graph_real_shape(
        width, height, seed=seed, model_id=model_id, vae=vae)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / graph_filename(width, height)

    torch.onnx.export(
        wrapper, (z, target), str(path),
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        opset_version=18, dynamo=True,
    )
    if fp16 or external_data:
        base.finalize(path, fp16=fp16, external_data=external_data)
    return path
