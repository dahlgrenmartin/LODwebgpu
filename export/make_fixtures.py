#!/usr/bin/env python3
"""Write browser harness fixtures into web/public/models.

`conda run` asserts on multi-line `python -c` arguments, so fixture generation
lives here rather than inline.

    python export/make_fixtures.py m0
    python export/make_fixtures.py adam
    python export/make_fixtures.py encoder
    python export/make_fixtures.py real
    python export/make_fixtures.py all
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "web" / "public" / "models"

# Reference FLUX.2 settings from dahlgrenmartin/LOD/src/LOD.py. Adam's beta/eps
# values are torch.optim.Adam defaults; the reference uses lr=0.03 for FLUX.2
# and 30 refinement steps.
REFERENCE_ADAM = {"lr": 0.03, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "steps": 30}
SEED = 5
DYNAMIC_RES = 16   # symbolic-capture example only; not an ORT supported size
REFERENCE_RES = 32 # 256x256 verification fixture
# Selectable checkpoint: FLUX.2, SD 1.5 and SDXL all wrap the same
# diffusers vae.Decoder, so one rewrite covers them. Override with
# LOD_MODEL_ID=madebyollin/sdxl-vae-fp16-fix python export/make_fixtures.py real
MODEL_ID = os.environ.get("LOD_MODEL_ID", "black-forest-labs/FLUX.2-small-decoder")

# Per-model optimizer settings. The FLUX.2 values come from the reference LOD
# implementation; SDXL's latent is 4 channels rather than 32, so the same step
# size does not transfer and its lr is a starting point, not a reference value.
ADAM_BY_MODEL = {
    "madebyollin/sdxl-vae-fp16-fix": {
        "lr": 0.03, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "steps": 30,
    },
}

# Exact ORT-Web image shapes, expressed as (width, height).
ORT_SIZES = [
    (256, 256),
    (512, 512),
    (768, 768),
    (1024, 768),
    (768, 1024),
    (768, 512),
    (512, 768),
]


def _export_wgsl_graph(*, real: bool) -> None:
    from export.graph_export import export_graph
    from export.to_onnx import build_graph_dynamic

    gm, wrapper, _, _ = build_graph_dynamic(
        res=DYNAMIC_RES, seed=SEED, real=real, model_id=MODEL_ID)
    export_graph(gm, wrapper, OUT)


def stage_m0() -> None:
    """Small structural fixture; kept square to make local smoke tests cheap."""
    from export.ort_shapes import graph_filename, write_manifest
    from export.to_onnx import export_joint, clear_outputs
    from export.reference import dump_reference

    OUT.mkdir(parents=True, exist_ok=True)
    clear_outputs(OUT)
    old = export_joint(
        res=DYNAMIC_RES, out_dir=OUT, fp16=True, external_data=True, seed=SEED)
    image = DYNAMIC_RES * 8
    old.rename(OUT / graph_filename(image, image))
    dump_reference(res=DYNAMIC_RES, out_dir=OUT, seed=SEED)
    _export_wgsl_graph(real=False)
    write_manifest(OUT, [(image, image)], REFERENCE_ADAM)
    print(f"m0 fixtures -> {OUT}")


def stage_adam() -> None:
    import numpy as np
    from export.adam_reference import adam_steps

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    n = 32 * DYNAMIC_RES * DYNAMIC_RES
    steps = REFERENCE_ADAM["steps"]
    z0 = rng.standard_normal(n).astype(np.float32)
    grads = [rng.standard_normal(n).astype(np.float32) for _ in range(steps)]
    cfg = {k: REFERENCE_ADAM[k] for k in ("lr", "beta1", "beta2", "eps")}
    expected = adam_steps(z0.copy(), grads, **cfg)
    (OUT / "adam_golden.json").write_text(json.dumps({
        "z0": z0.tolist(), "grads": [g.tolist() for g in grads],
        "expected": expected.tolist(), "cfg": cfg,
    }))
    print(f"adam golden -> {OUT / 'adam_golden.json'}")


def stage_real() -> None:
    """Everything the demo needs, built from the selected VAE checkpoint."""
    from export.ort_shapes import export_joint_real_shape, write_manifest
    from export.to_onnx import export_encoder, clear_outputs, load_real_vae
    from export.reference import dump_reference_real

    OUT.mkdir(parents=True, exist_ok=True)
    clear_outputs(OUT)

    # Load the checkpoint once for all seven static captures. Each exported graph
    # still gets its own shape-specialized activation plan, while model weights
    # are sourced from the same in-memory module during generation.
    vae = load_real_vae(MODEL_ID)
    for width, height in ORT_SIZES:
        export_joint_real_shape(
            width, height, OUT,
            fp16=True, external_data=True, seed=SEED,
            model_id=MODEL_ID, vae=vae,
        )
        print(f"ORT graph -> {width}x{height}")

    # The verification harness only needs one golden trajectory; use the first
    # supported square size so it can run against the first manifest entry.
    # Must use the SAME checkpoint as everything else: its default is FLUX.2,
    # so omitting model_id silently produces golden tensors from a different
    # model - which only surfaces as a latent-channel mismatch at run time.
    dump_reference_real(res=REFERENCE_RES, out_dir=OUT, seed=SEED,
                        model_id=MODEL_ID)
    export_encoder(OUT, model_id=MODEL_ID, fp16=True, external_data=True)
    _export_wgsl_graph(real=True)
    # Each static export appends its own copy of the weights, so collapse them
    # to one shared set and shard it under the 100 MB per-file limit.
    from export.to_onnx import dedupe_external_data
    report = dedupe_external_data(OUT)
    print(f"weights {report['before_mb']:.1f} MB -> "
          f"{report['shards']} = {report['shard_mb']} MB")
    write_manifest(OUT, ORT_SIZES, ADAM_BY_MODEL.get(MODEL_ID, REFERENCE_ADAM),
                   latent_channels=vae.config.latent_channels,
                   weights=report["shards"], model=MODEL_ID)
    print(f"real-checkpoint fixtures -> {OUT}  "
          f"[{MODEL_ID}, latent_channels={vae.config.latent_channels}]")


def stage_encoder() -> None:
    from export.to_onnx import export_encoder
    OUT.mkdir(parents=True, exist_ok=True)
    export_encoder(OUT)
    print(f"encoder -> {OUT / 'encoder.onnx'}")


STAGES = {"m0": stage_m0, "adam": stage_adam, "encoder": stage_encoder,
          "real": stage_real}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES) + ["all"])
    args = ap.parse_args()
    for name in (sorted(STAGES) if args.stage == "all" else [args.stage]):
        STAGES[name]()