"""not1d against the original ot1d, on the same inputs. Skipped when ot1d is
not installed: it ships only an sdist, see the ot1d job in CI."""

import numpy as np
import pytest

from not1d import ot1d

OT1D = pytest.importorskip("OT1D").OT1D

rng = np.random.default_rng(7)


@pytest.mark.parametrize("m,n", [(1, 1), (10, 10), (10, 7), (1000, 333), (20_000, 50_000)])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_same_as_ot1d(m, n, p, weighted):
    x, y = rng.normal(size=m), rng.uniform(-1, 3, size=n)
    if weighted:
        mu, nu = rng.uniform(size=m), rng.uniform(size=n)
        mu, nu = mu / mu.sum(), nu / nu.sum()
        expected = OT1D(x, y, mu, nu, p=p, threads=1)
        assert ot1d(x, y, mu, nu, p=p) == pytest.approx(expected, rel=1e-9)
    else:
        expected = OT1D(x, y, p=p, threads=1)
        assert ot1d(x, y, p=p) == pytest.approx(expected, rel=1e-9)
