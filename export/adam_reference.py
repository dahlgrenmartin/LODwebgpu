"""Reference Adam, shared by the Python tests and the WGSL kernel's golden data."""
from __future__ import annotations

import numpy as np


def adam_steps(z, grads, lr: float, beta1: float, beta2: float, eps: float):
    """Bias-corrected Adam, matching torch.optim.Adam defaults (no weight decay)."""
    z = np.asarray(z, dtype=np.float32).copy()
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    for t, g in enumerate(grads, start=1):
        g = np.asarray(g, dtype=np.float32)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1 ** t)
        vhat = v / (1.0 - beta2 ** t)
        z = z - lr * mhat / (np.sqrt(vhat) + eps)
    return z.astype(np.float32)
