"""Serialize the rewritten joint graph for the WGSL interpreter.

This is the second execution backend, alongside the ONNX/ORT-Web one; neither
replaces the other.

The point of this format is that nothing about the image size is baked in.  The
graph is captured with symbolic H/W, and any integer argument that depends on
those symbols is emitted as a small expression tree.  The interpreter binds the
symbols from the actual input tensor at run time and evaluates the trees, which
is how PyTorch manages to accept any size and what LOD requires.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path

import numpy as np
import torch

# --- symbolic integer serialization ----------------------------------------


def _expr_tree(expr):
    """Convert a sympy expression over shape symbols into a JSON tree."""
    import sympy

    if isinstance(expr, (int, np.integer)):
        return int(expr)
    if expr.is_Integer:
        return int(expr)
    if expr.is_Symbol:
        return {"sym": str(expr)}
    if expr.is_Add:
        return {"op": "add", "args": [_expr_tree(a) for a in expr.args]}
    if expr.is_Mul:
        return {"op": "mul", "args": [_expr_tree(a) for a in expr.args]}
    if expr.is_Pow:
        return {"op": "pow", "args": [_expr_tree(expr.base), _expr_tree(expr.exp)]}
    name = type(expr).__name__
    if name in ("FloorDiv", "floor"):
        return {"op": "floordiv", "args": [_expr_tree(a) for a in expr.args]}
    if name == "Mod":
        return {"op": "mod", "args": [_expr_tree(a) for a in expr.args]}
    raise NotImplementedError(f"cannot serialize symbolic expression {expr!r} ({name})")


def _int_arg(v):
    """An int, or an expression tree when the value is symbolic."""
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, bool):
        return bool(v)
    node = getattr(v, "node", None)
    expr = getattr(node, "expr", None)
    if expr is None:
        raise NotImplementedError(f"cannot serialize integer argument {v!r}")
    return _expr_tree(expr)


def _shape_list(shape):
    return [_int_arg(d) for d in shape]


def free_symbols(shape) -> set:
    out = set()
    for d in shape:
        node = getattr(d, "node", None)
        expr = getattr(node, "expr", None)
        if expr is not None:
            out |= {str(s) for s in expr.free_symbols}
    return out


# --- graph walk -------------------------------------------------------------


# Metadata assertions carry no computation; the interpreter has nothing to do
# with them and they would only need stub kernels.
DROPPED_OPS = {"aten::_assert_tensor_metadata"}


def opname(node) -> str | None:
    if node.op != "call_function":
        return None
    t = node.target
    if t is operator.getitem:
        return "getitem"
    if hasattr(t, "_schema"):
        s = t._schema
        return str(s.name) + (("." + s.overload_name) if s.overload_name else "")
    return str(t)


def _encode_arg(a, name_of):
    """Encode a node argument: tensor ref, int/expr, list, or literal."""
    import torch.fx as fx

    if isinstance(a, fx.Node):
        return {"ref": name_of[a]}
    if isinstance(a, (list, tuple)):
        return [_encode_arg(x, name_of) for x in a]
    if a is None or isinstance(a, (bool, str, float)):
        return a
    if isinstance(a, torch.dtype):
        return {"dtype": str(a).replace("torch.", "")}
    if isinstance(a, torch.device):
        return {"device": str(a)}
    if isinstance(a, torch.layout):
        return {"layout": str(a)}
    if isinstance(a, torch.memory_format):
        return {"memory_format": str(a)}
    return _int_arg(a)


def export_graph(gm, wrapper, out_dir: Path, name: str = "lod_graph") -> Path:
    """Write <name>.json plus <name>.weights.bin."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name_of: dict = {}
    for i, n in enumerate(gm.graph.nodes):
        name_of[n] = n.name or f"n{i}"

    # Constants come from the wrapper's buffers, in placeholder order.
    plan = wrapper._plan
    buffers = dict(wrapper.named_buffers())

    inputs, constants, chunks, offset = [], [], [], 0
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(plan):
        raise RuntimeError(f"placeholder/plan mismatch: {len(placeholders)} vs {len(plan)}")

    symbols: set = set()
    for n, (kind, key) in zip(placeholders, plan):
        val = n.meta.get("val")
        shape = _shape_list(val.shape) if val is not None else None
        if val is not None:
            symbols |= free_symbols(val.shape)
        if kind in ("z", "target"):
            inputs.append({"name": kind, "node": name_of[n], "shape": shape})
        else:
            arr = buffers[key].detach().numpy().ravel().astype("<f4")
            constants.append({"node": name_of[n], "offset": offset,
                              "numel": int(arr.size), "shape": shape})
            chunks.append(arr)
            offset += int(arr.size)

    nodes = []
    unsupported = set()
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        op = opname(n)
        if op in DROPPED_OPS:
            continue
        val = n.meta.get("val")
        entry = {
            "name": name_of[n],
            "op": op,
            "args": [_encode_arg(a, name_of) for a in n.args],
        }
        if n.kwargs:
            entry["kwargs"] = {k: _encode_arg(v, name_of) for k, v in n.kwargs.items()}
        if val is not None and hasattr(val, "shape"):
            entry["shape"] = _shape_list(val.shape)
            symbols |= free_symbols(val.shape)
        nodes.append(entry)
        if op:
            unsupported.add(op)

    out_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    outputs = [name_of[a] if hasattr(a, "name") else None for a in out_node.args[0]]

    blob = out_dir / f"{name}.weights.bin"
    blob.write_bytes(np.concatenate(chunks).tobytes() if chunks else b"")

    doc = {
        "symbols": sorted(symbols),
        "inputs": inputs,
        "constants": constants,
        "weights": blob.name,
        "nodes": nodes,
        "outputs": [{"name": nm, "label": lbl} for nm, lbl in
                    zip(outputs, ["loss", "pred", "score", "grad_z"])],
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def op_inventory(gm) -> dict:
    """Counts of every call_function op, for planning kernel coverage."""
    import collections
    return dict(collections.Counter(
        opname(n) for n in gm.graph.nodes if opname(n)))
