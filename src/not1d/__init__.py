"""Optimal transport in 1D, a minimal Nim port of ot1d built with nimlang."""

import numpy as np

from not1d.core import ot1d as _ot1d, ot1dWeighted as _ot1d_weighted

__all__ = ["ot1d"]


def ot1d(x, y, mu=None, nu=None, p=1, sorting=True):
    """Kantorovich-Wasserstein distance of order p (1 or 2) between two
    discrete measures on the real line.

    x, y: support points (1D numpy arrays or lists).
    mu, nu: masses of the points, with the same total; None for uniform masses.
    sorting: False if x and y (with their masses) are already sorted.
    """
    # Arguments are checked here as well as in Nim: nimpy turns a Nim
    # ValueError into nimpy.ValueError, which is not a Python ValueError.
    if p not in (1, 2):
        raise ValueError(f"p must be 1 or 2, got {p}")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must be non-empty 1D arrays")
    if mu is None and nu is None:
        return _ot1d(x, y, p, sorting)
    mu = np.full(len(x), 1 / len(x)) if mu is None else np.asarray(mu, dtype=np.float64)
    nu = np.full(len(y), 1 / len(y)) if nu is None else np.asarray(nu, dtype=np.float64)
    if mu.shape != x.shape or nu.shape != y.shape:
        raise ValueError("mu and nu must have the length of x and y")
    return _ot1d_weighted(x, y, mu, nu, p, sorting)
