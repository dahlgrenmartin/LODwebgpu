"""Dump golden tensors for the browser parity harness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from export.to_onnx import build_graph, build_graph_real


def dump_reference(res: int, out_dir: Path, seed: int = 5) -> Path:
    """Write reference_<image>.bin (raw little-endian fp32) plus a .json index."""
    _, wrapper, z, t = build_graph(res=res, seed=seed)
    with torch.no_grad():
        loss, pred, score, grad_z = wrapper(z, t)

    return _write_named(z, t, loss, pred, score, grad_z, res, out_dir)


def _write(wrapper, z, t, res: int, out_dir: Path) -> Path:
    with torch.no_grad():
        loss, pred, score, grad_z = wrapper(z, t)
    return _write_named(z, t, loss, pred, score, grad_z, res, out_dir)


def _write_named(z, t, loss, pred, score, grad_z, res: int, out_dir: Path) -> Path:
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


def dump_reference_real(res: int, out_dir: Path, seed: int = 5,
                        model_id: str = "black-forest-labs/FLUX.2-small-decoder") -> Path:
    """Golden tensors from the trained checkpoint, same layout as dump_reference."""
    _, wrapper, z, t = build_graph_real(res=res, seed=seed, model_id=model_id)
    return _write(wrapper, z, t, res, out_dir)
