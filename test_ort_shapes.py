#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from export.make_fixtures import ORT_SIZES
from export.ort_shapes import graph_filename, write_manifest


ADAM = {"lr": 0.03, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "steps": 30}


def test_requested_ort_sizes_are_exact():
    assert ORT_SIZES == [
        (256, 256),
        (512, 512),
        (768, 768),
        (1024, 768),
        (768, 1024),
        (768, 512),
        (512, 768),
    ]


def test_graph_filename_uses_width_height():
    assert graph_filename(1024, 768) == "lod_joint_1024x768.onnx"
    assert graph_filename(768, 1024) == "lod_joint_768x1024.onnx"


def test_manifest_records_rectangular_shapes():
    with tempfile.TemporaryDirectory() as td:
        path = write_manifest(Path(td), [(256, 256), (1024, 768)], ADAM)
        manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["resolutions"] == [
        {
            "width": 256,
            "height": 256,
            "latent": [1, 32, 32, 32],
            "numel": 32 * 32 * 32,
            "graph": "lod_joint_256x256.onnx",
        },
        {
            "width": 1024,
            "height": 768,
            "latent": [1, 32, 96, 128],
            "numel": 32 * 96 * 128,
            "graph": "lod_joint_1024x768.onnx",
        },
    ]


def test_manifest_rejects_non_multiple_of_eight():
    with tempfile.TemporaryDirectory() as td:
        try:
            write_manifest(Path(td), [(257, 256)], ADAM)
        except ValueError as e:
            assert "multiple of 8" in str(e)
        else:
            raise AssertionError("expected non-multiple-of-8 size to be rejected")


if __name__ == "__main__":
    test_requested_ort_sizes_are_exact()
    test_graph_filename_uses_width_height()
    test_manifest_records_rectangular_shapes()
    test_manifest_rejects_non_multiple_of_eight()
    print("ORT shape export tests passed")
