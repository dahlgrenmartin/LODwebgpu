#!/usr/bin/env python3
"""Regression tests for generated browser model assets."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from export import make_fixtures
from export.to_onnx import clear_outputs


def test_clear_outputs_removes_wgsl_graph_pair():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        graph = out / "lod_graph.json"
        weights = out / "lod_graph.weights.bin"
        graph.write_text('{"weights":"lod_graph.weights.bin"}', encoding="utf-8")
        weights.write_bytes(b"\x00\x00\x00\x00")

        clear_outputs(out)

        assert not graph.exists(), "stale lod_graph.json survives after its weights are deleted"
        assert not weights.exists()


def test_real_stage_regenerates_wgsl_graph():
    sentinel_gm = object()
    sentinel_wrapper = object()

    with tempfile.TemporaryDirectory() as td, \
         mock.patch.object(make_fixtures, "OUT", Path(td)), \
         mock.patch("export.to_onnx.clear_outputs"), \
         mock.patch("export.to_onnx.export_joint_real"), \
         mock.patch("export.to_onnx.export_encoder"), \
         mock.patch("export.to_onnx.write_manifest"), \
         mock.patch("export.to_onnx.build_graph_dynamic", return_value=(sentinel_gm, sentinel_wrapper, None, None)) as build_dynamic, \
         mock.patch("export.reference.dump_reference_real"), \
         mock.patch("export.graph_export.export_graph") as export_graph:
        make_fixtures.stage_real()

    build_dynamic.assert_called_once()
    export_graph.assert_called_once_with(sentinel_gm, sentinel_wrapper, Path(td))


if __name__ == "__main__":
    test_clear_outputs_removes_wgsl_graph_pair()
    test_real_stage_regenerates_wgsl_graph()
    print("fixture pipeline tests passed")
