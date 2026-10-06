"""Time not1d against scipy and, when installed, the original ot1d.

    uv run python benchmark.py

ot1d ships only an sdist that needs Cython and numpy already installed:
    uv pip install numpy cython setuptools && uv pip install --no-build-isolation ot1d
"""

import timeit

import numpy as np
from scipy.stats import wasserstein_distance

from not1d import ot1d

try:
    from OT1D import OT1D
except ImportError:
    OT1D = None

rng = np.random.default_rng(13)


def best(f, number):
    return min(timeit.repeat(f, number=number, repeat=5)) / number


print("| n | case | not1d | scipy | ot1d (1 thread) | ot1d (8 threads) |")
print("|--:|:-----|------:|------:|----------------:|-----------------:|")
for n in [1_000, 100_000, 1_000_000]:
    number = max(1, 100_000 // n)
    x, y = rng.uniform(1, 2, n), rng.uniform(0, 1, n // 2)
    mu, nu = rng.uniform(size=n), rng.uniform(size=n // 2)
    mu, nu = mu / mu.sum(), nu / nu.sum()
    for case, args in [("uniform", ()), ("weighted", (mu, nu))]:
        row = [
            best(lambda: ot1d(x, y, *args), number),
            best(lambda: wasserstein_distance(x, y, *args), number),
        ]
        if OT1D is not None:
            row += [best(lambda: OT1D(x, y, *args, threads=t), number) for t in (1, 8)]
        cells = " | ".join(f"{1e3 * t:.2f} ms" for t in row)
        print(f"| {n:,} | {case} | {cells} |")
