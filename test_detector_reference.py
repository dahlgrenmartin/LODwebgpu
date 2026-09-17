#!/usr/bin/env python3
"""Reference-semantics tests for the LOD wavelet detector."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import flux2_lod_aot_probe as probe


def test_level3_detector_is_lh_hl_energy():
    det = probe.Sym4Level3Detector(channels=1)
    h = torch.full((1, 1, 2, 2), 2.0)
    v = torch.full((1, 1, 2, 2), 4.0)
    d = torch.full((1, 1, 2, 2), 100.0)
    det.bands = lambda x, bank=None: (h, v, d)

    energy = float(det(torch.zeros(1, 1, 2, 2)))
    expected = float(0.5 * ((h * h).mean() + (v * v).mean()))

    assert math.isclose(energy, expected, rel_tol=0.0, abs_tol=1e-6)


def test_joint_detector_receives_residual():
    class IdentityDecoder(nn.Module):
        def forward(self, z):
            return z

    class MeanDetector(nn.Module):
        def forward(self, x):
            return x.mean()

    m = probe.JointLOD.__new__(probe.JointLOD)
    nn.Module.__init__(m)
    m.decoder = IdentityDecoder()
    m.loss_kind = "l1"
    m.sym4 = nn.Identity()
    m.detector = MeanDetector()

    z = torch.tensor([[[[2.0]], [[4.0]], [[6.0]]]])
    target = torch.ones_like(z)
    _, pred, score = m(z, target)

    residual = pred - target
    assert torch.allclose(score, residual.mean())


def test_flux2_reference_optimizer_config():
    from export import make_fixtures

    assert make_fixtures.REFERENCE_ADAM == {
        "lr": 0.03,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "steps": 30,
    }


if __name__ == "__main__":
    test_level3_detector_is_lh_hl_energy()
    test_joint_detector_receives_residual()
    test_flux2_reference_optimizer_config()
    print("detector reference tests passed")
