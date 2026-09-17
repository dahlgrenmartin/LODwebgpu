"""Wrap a lifted joint GraphModule so ONNX sees initializers, not placeholders."""
from __future__ import annotations

import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """Exposes forward(z, target); every parameter/buffer becomes an initializer.

    aot_export_module lifts parameters into graph placeholders.  Exporting the
    GraphModule directly would therefore produce an ONNX model with one input per
    parameter.  Registering them here as buffers makes the exporter fold them into
    initializers instead, leaving exactly two real inputs.
    """

    def __init__(self, gm, sig, module: nn.Module):
        super().__init__()
        self.gm = gm
        state = module.state_dict()
        self._plan: list[tuple[str, str]] = []   # (kind, buffer name)
        for n in gm.graph.nodes:
            if n.op != "placeholder":
                continue
            if n.name in sig.inputs_to_parameters:
                self._plan.append(
                    ("const", self._store(state[sig.inputs_to_parameters[n.name]], n.name)))
            elif n.name in sig.inputs_to_buffers:
                self._plan.append(
                    ("const", self._store(state[sig.inputs_to_buffers[n.name]], n.name)))
            elif n.name == sig.user_inputs[0]:
                self._plan.append(("z", ""))
            elif n.name == sig.user_inputs[1]:
                self._plan.append(("target", ""))
            else:
                raise KeyError(f"unclassified placeholder {n.name}")

    def _store(self, tensor: torch.Tensor, name: str) -> str:
        key = "c_" + name.replace(".", "_")
        self.register_buffer(key, tensor.detach().clone(), persistent=True)
        return key

    def forward(self, z, target):
        args = []
        for kind, key in self._plan:
            if kind == "z":
                args.append(z)
            elif kind == "target":
                args.append(target)
            else:
                args.append(getattr(self, key))
        return self.gm(*args)
