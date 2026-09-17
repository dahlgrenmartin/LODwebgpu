#!/usr/bin/env python3
"""Per-node statistics for the rewritten graph, to localize interpreter bugs.

Running the whole graph and comparing four outputs tells you something is wrong;
it does not tell you where. This records mean and max-abs for every node so the
browser can find the FIRST node that diverges, which is the one to look at.

Deliberately small: the graph is shape-agnostic, so a bug reproduces at 64x64
where a readback per node is affordable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.fx as fx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class StatRecorder(fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.stats: dict[str, dict] = {}

    def run_node(self, n):
        out = super().run_node(n)
        if n.op == "call_function" and isinstance(out, torch.Tensor):
            a = out.detach().float()
            self.stats[n.name] = {
                "mean": float(a.mean()), "absmax": float(a.abs().max()),
                "numel": int(a.numel()), "shape": list(a.shape),
            }
        return out


def write(out_dir: Path, res: int = 16, seed: int = 5) -> Path:
    from export.to_onnx import build_graph_dynamic

    gm, wrapper, _, _ = build_graph_dynamic(res=res, seed=seed, real=True)

    # The captured graph's generated code still refers to its trace-time shape
    # symbols, so it only runs at the size it was captured at. build_graph_dynamic
    # deliberately uses different height and width, so this also exercises a
    # non-square input.
    res_h, res_w = res, res + 8
    torch.manual_seed(seed)
    z = torch.randn(1, 32, res_h, res_w)
    t = torch.randn(1, 3, res_h * 8, res_w * 8)

    # Feed the graph in placeholder order, as the wrapper does.
    args = []
    for kind, key in wrapper._plan:
        if kind == "z":
            args.append(z)
        elif kind == "target":
            args.append(t)
        else:
            args.append(getattr(wrapper, key))

    rec = StatRecorder(gm)
    with torch.no_grad():
        outs = rec.run(*args)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob, meta, offset = [], {}, 0
    for name, arr in (("z", z), ("target", t),
                      ("loss", outs[0]), ("pred", outs[1]),
                      ("score", outs[2]), ("grad_z", outs[3])):
        a = np.asarray(arr.detach().numpy()).ravel().astype("<f4")
        meta[name] = {"offset": offset, "count": int(a.size),
                      "shape": list(arr.shape)}
        blob.append(a)
        offset += int(a.size)

    (out_dir / "trace.bin").write_bytes(np.concatenate(blob).tobytes())
    (out_dir / "trace.json").write_text(json.dumps({
        "latent": [res_h, res_w], "image": [res_h * 8, res_w * 8],
        "tensors": meta, "nodes": rec.stats,
    }), encoding="utf-8")
    return out_dir / "trace.json"


if __name__ == "__main__":
    p = write(ROOT / "web" / "public" / "models")
    doc = json.loads(p.read_text())
    print(f"{p.name}: {len(doc['nodes'])} node stats at {doc['image']}")
