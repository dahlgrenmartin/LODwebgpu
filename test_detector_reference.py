#!/usr/bin/env python3
"""Reference-semantics tests for the LOD wavelet detector."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import flux2_lod_aot_probe as probe


def test_level1_loss_sums_three_detail_band_means():
    loss_fn = probe.Sym4Level1Loss()
    residual = torch.arange(1 * 3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16) / 255.0
    x = probe.zero_pad2d(residual, 6, 6, 6, 6)
    d = F.conv2d(x, loss_fn.kernels, stride=2, groups=3)
    expected = sum(d[:, band::3].abs().mean() for band in range(3))

    got = loss_fn(residual)
    assert torch.allclose(got, expected, rtol=1e-6, atol=1e-7)


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
    test_level1_loss_sums_three_detail_band_means()
    test_level3_detector_is_lh_hl_energy()
    test_joint_detector_receives_residual()
    test_flux2_reference_optimizer_config()
    print("detector reference tests passed")
