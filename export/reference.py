"""Dump golden tensors for the browser parity harness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from export.to_onnx import build_graph


def dump_reference(res: int, out_dir: Path, seed: int = 5) -> Path:
    """Write reference_<image>.bin (raw little-endian fp32) plus a .json index."""
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
