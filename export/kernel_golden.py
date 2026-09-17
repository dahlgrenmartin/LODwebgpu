#!/usr/bin/env python3
"""Golden data for the WGSL kernel unit tests.

Each case names the kernel, its parameters, the inputs, and the result PyTorch
produces. The browser test runs the same case on the GPU and compares.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _t(x):
    return {"shape": list(x.shape), "data": x.flatten().tolist()}


def build_cases() -> list[dict]:
    torch.manual_seed(20)
    cases: list[dict] = []

    # --- gather: permute / expand / slice --------------------------------
    a = torch.randn(2, 3, 4)
    cases.append({"kernel": "gather", "name": "permute(2,0,1)",
                  "src": _t(a), "perm": [2, 0, 1],
                  "expected": _t(a.permute(2, 0, 1).contiguous())})
    b = torch.randn(1, 3, 1)
    cases.append({"kernel": "gather", "name": "expand to (4,3,5)",
                  "src": _t(b), "expand": [4, 3, 5],
                  "expected": _t(b.expand(4, 3, 5).contiguous())})
    c = torch.randn(5, 6)
    cases.append({"kernel": "gather", "name": "slice rows 1:4",
                  "src": _t(c), "slice": {"dim": 0, "start": 1, "end": 4},
                  "expected": _t(c[1:4].contiguous())})

    # --- elementwise ------------------------------------------------------
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    for op, fn in (("add", torch.add), ("sub", torch.sub),
                   ("mul", torch.mul), ("div", torch.div)):
        cases.append({"kernel": "elementwise", "name": f"{op} same-shape",
                      "op": op, "a": _t(x), "b": _t(y), "expected": _t(fn(x, y))})
    bc = torch.randn(1, 3, 1)
    cases.append({"kernel": "elementwise", "name": "mul broadcast (1,3,1)",
                  "op": "mul", "a": _t(x), "b": _t(bc), "expected": _t(x * bc)})
    rank = torch.randn(4)
    cases.append({"kernel": "elementwise", "name": "add rank-broadcast (4,)",
                  "op": "add", "a": _t(x), "b": _t(rank), "expected": _t(x + rank)})
    for op, fn in (("abs", torch.abs), ("neg", torch.neg),
                   ("sigmoid", torch.sigmoid), ("silu", F.silu)):
        cases.append({"kernel": "elementwise", "name": f"{op} unary",
                      "op": op, "a": _t(x), "b": None, "scalar": 0.0,
                      "expected": _t(fn(x))})
    cases.append({"kernel": "elementwise", "name": "add scalar 0.75",
                  "op": "add", "a": _t(x), "b": None, "scalar": 0.75,
                  "expected": _t(x + 0.75)})
    cases.append({"kernel": "elementwise", "name": "gt scalar 0",
                  "op": "gt", "a": _t(x), "b": None, "scalar": 0.0,
                  "expected": _t((x > 0).to(torch.float32))})
    cases.append({"kernel": "elementwise", "name": "lt scalar 0",
                  "op": "lt", "a": _t(x), "b": None, "scalar": 0.0,
                  "expected": _t((x < 0).to(torch.float32))})
    cases.append({"kernel": "elementwise", "name": "rsub 1 - x",
                  "op": "rsub", "a": _t(x), "b": None, "scalar": 1.0,
                  "expected": _t(1.0 - x)})

    # --- reduce -----------------------------------------------------------
    r = torch.randn(2, 5, 3)
    for dims in ([2], [1], [0], [1, 2]):
        cases.append({"kernel": "reduce", "name": f"sum dims={dims}",
                      "dims": dims, "mean": False, "src": _t(r),
                      "expected": _t(r.sum(dim=dims, keepdim=True))})
        cases.append({"kernel": "reduce", "name": f"mean dims={dims}",
                      "dims": dims, "mean": True, "src": _t(r),
                      "expected": _t(r.mean(dim=dims, keepdim=True))})

    # --- convolution ------------------------------------------------------
    def conv_case(name, xs, ws, stride, pad, dil, groups, transposed):
        x = torch.randn(*xs)
        w = torch.randn(*ws)
        y = torch.ops.aten.convolution(
            x, w, None, list(stride), list(pad), list(dil), transposed,
            [0, 0], groups)
        return {"kernel": "conv2d", "name": name, "src": _t(x), "weight": _t(w),
                "stride": list(stride), "padding": list(pad), "dilation": list(dil),
                "groups": groups, "transposed": transposed,
                "expected": _t(y)}

    cases.append(conv_case("conv3x3 pad1", (1, 4, 7, 9), (6, 4, 3, 3),
                           (1, 1), (1, 1), (1, 1), 1, False))
    cases.append(conv_case("conv1x1", (1, 5, 4, 6), (3, 5, 1, 1),
                           (1, 1), (0, 0), (1, 1), 1, False))
    cases.append(conv_case("conv grouped stride2 k8", (1, 3, 28, 30), (9, 1, 8, 8),
                           (2, 2), (0, 0), (1, 1), 3, False))
    cases.append(conv_case("convT 3x3 pad1", (1, 6, 7, 9), (6, 4, 3, 3),
                           (1, 1), (1, 1), (1, 1), 1, True))
    cases.append(conv_case("convT grouped stride2", (1, 9, 11, 12), (9, 1, 8, 8),
                           (2, 2), (0, 0), (1, 1), 3, True))

    # --- group norm -------------------------------------------------------
    for affine in (True, False):
        N, C, H, W, G = 1, 8, 5, 4, 4
        x = torch.randn(N, C, H, W)
        wt = torch.randn(C) if affine else None
        bs = torch.randn(C) if affine else None
        out, mean, rstd = torch.ops.aten.native_group_norm(
            x, wt, bs, N, C, H * W, G, 1e-5)
        cases.append({"kernel": "groupnorm",
                      "name": f"groupnorm affine={affine}",
                      "src": _t(x), "N": N, "C": C, "HxW": H * W, "G": G,
                      "eps": 1e-5,
                      "weight": _t(wt) if affine else None,
                      "bias": _t(bs) if affine else None,
                      "expected": _t(out), "mean": _t(mean), "rstd": _t(rstd)})

    # --- matmul -----------------------------------------------------------
    a2 = torch.randn(4, 5); b2 = torch.randn(5, 3)
    cases.append({"kernel": "matmul", "name": "mm", "a": _t(a2), "b": _t(b2),
                  "B": 1, "M": 4, "K": 5, "N": 3, "expected": _t(a2 @ b2)})
    ab = torch.randn(2, 4, 5); bb = torch.randn(2, 5, 3)
    cases.append({"kernel": "matmul", "name": "bmm", "a": _t(ab), "b": _t(bb),
                  "B": 2, "M": 4, "K": 5, "N": 3, "expected": _t(torch.bmm(ab, bb))})
    bias = torch.randn(3)
    cases.append({"kernel": "matmul", "name": "addmm row-vector bias",
                  "a": _t(a2), "b": _t(b2), "bias": _t(bias), "biasIsRowVector": True,
                  "B": 1, "M": 4, "K": 5, "N": 3,
                  "expected": _t(torch.addmm(bias, a2, b2))})

    # --- softmax ----------------------------------------------------------
    sm = torch.randn(2, 6, 7)
    for dim in (-1, 1):
        cases.append({"kernel": "softmax", "name": f"softmax dim={dim}",
                      "src": _t(sm), "dim": dim,
                      "expected": _t(torch.softmax(sm, dim=dim))})
    return cases



def write(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "kernel_golden.json"
    p.write_text(json.dumps(build_cases()), encoding="utf-8")
    return p


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    p = write(root / "web" / "public" / "models")
    print(f"{p} ({p.stat().st_size / 1024:.0f} KB, {len(build_cases())} cases)")
