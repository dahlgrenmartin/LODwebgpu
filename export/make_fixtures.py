#!/usr/bin/env python3
"""Write browser harness fixtures into web/public/models.

`conda run` asserts on multi-line `python -c` arguments, so fixture generation
lives here rather than inline.

    python export/make_fixtures.py m0
    python export/make_fixtures.py adam
    python export/make_fixtures.py encoder
    python export/make_fixtures.py all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "web" / "public" / "models"

# Fixture hyperparameters.  NOT the reference LOD values -- see the plan's
# Global Constraints.  They exist so manifest validation passes.
FIXTURE_ADAM = {"lr": 0.05, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "steps": 30}
SEED = 5
RES = 16   # latent 16 -> 128x128 image
MODEL_ID = "black-forest-labs/FLUX.2-small-decoder"


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


def stage_real() -> None:
    """Everything the demo needs, built from the trained FLUX.2-small checkpoint."""
    from export.to_onnx import export_joint_real, export_encoder, write_manifest
    from export.reference import dump_reference_real
    OUT.mkdir(parents=True, exist_ok=True)
    export_joint_real(res=RES, out_dir=OUT, fp16=True, external_data=True, seed=SEED)
    dump_reference_real(res=RES, out_dir=OUT, seed=SEED)
    export_encoder(OUT, model_id=MODEL_ID, fp16=True, external_data=True)
    write_manifest(OUT, [RES], FIXTURE_ADAM)
    print(f"real-checkpoint fixtures -> {OUT}")


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
