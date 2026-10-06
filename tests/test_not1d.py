import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from not1d import ot1d


def reference(x, y, mu=None, nu=None, p=1):
    """W_p from the quantile functions: integral over t in [0, 1] of
    |F^-1(t) - G^-1(t)|^p, exact on the merged breakpoints of the two CDFs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mu = np.full(len(x), 1 / len(x)) if mu is None else np.asarray(mu, float)
    nu = np.full(len(y), 1 / len(y)) if nu is None else np.asarray(nu, float)
    ix, iy = np.argsort(x), np.argsort(y)
    x, mu, y, nu = x[ix], mu[ix], y[iy], nu[iy]
    cx, cy = np.cumsum(mu), np.cumsum(nu)
    t = np.union1d(cx, cy)
    t = t[t <= min(cx[-1], cy[-1])]
    dt = np.diff(t, prepend=0.0)
    mid = t - dt / 2
    qx = x[np.minimum(np.searchsorted(cx, mid), len(x) - 1)]
    qy = y[np.minimum(np.searchsorted(cy, mid), len(y) - 1)]
    return np.sum(dt * np.abs(qx - qy) ** p) ** (1 / p)


rng = np.random.default_rng(13)
SIZES = [(1, 1), (10, 10), (10, 7), (1000, 333), (500, 1000)]


@pytest.mark.parametrize("m,n", SIZES)
def test_p1_matches_scipy(m, n):
    x, y = rng.normal(size=m), rng.uniform(-1, 3, size=n)
    assert ot1d(x, y) == pytest.approx(wasserstein_distance(x, y))


@pytest.mark.parametrize("m,n", SIZES)
def test_p1_weighted_matches_scipy(m, n):
    x, y = rng.normal(size=m), rng.uniform(-1, 3, size=n)
    mu, nu = rng.uniform(size=m), rng.uniform(size=n)
    mu, nu = mu / mu.sum(), nu / nu.sum()
    expected = wasserstein_distance(x, y, mu, nu)
    assert ot1d(x, y, mu, nu) == pytest.approx(expected)


@pytest.mark.parametrize("m,n", SIZES)
@pytest.mark.parametrize("weighted", [False, True])
def test_p2_matches_reference(m, n, weighted):
    x, y = rng.normal(size=m), rng.uniform(-1, 3, size=n)
    mu = nu = None
    if weighted:
        mu, nu = rng.uniform(size=m), rng.uniform(size=n)
        mu, nu = mu / mu.sum(), nu / nu.sum()
    assert ot1d(x, y, mu, nu, p=2) == pytest.approx(reference(x, y, mu, nu, p=2))


def test_inputs_are_not_modified():
    x, y = rng.normal(size=100), rng.normal(size=50)
    x0, y0 = x.copy(), y.copy()
    ot1d(x, y, p=2)
    ot1d(x, y, np.full(100, 0.01), np.full(50, 0.02))
    assert (x == x0).all() and (y == y0).all()


def test_lists_and_strided_arrays():
    x, y = rng.normal(size=20), rng.normal(size=9)
    expected = wasserstein_distance(x[::2], y)
    assert ot1d(x[::2], y) == pytest.approx(expected)
    assert ot1d(list(x[::2]), list(y)) == pytest.approx(expected)


def test_presorted():
    x, y = np.sort(rng.normal(size=30)), np.sort(rng.normal(size=20))
    assert ot1d(x, y, sorting=False) == pytest.approx(ot1d(x, y))


def test_known_value():
    # Moving every point by 1 costs 1, whatever the order.
    assert ot1d([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], p=1) == pytest.approx(1.0)
    assert ot1d([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], p=2) == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs", [dict(p=3), dict(mu=[1.0])])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        ot1d([0.0, 1.0], [2.0], **kwargs)


@pytest.mark.parametrize("m,n", [(5000, 3000), (2000, 2000)])
def test_ties_and_wide_range(m, n):
    # Many equal points, and magnitudes from 1e-300 to 1e300 of both signs:
    # exercises every digit of the radix sort.
    x = rng.integers(-5, 5, size=m).astype(float)
    y = rng.choice([-1e300, -1.0, -1e-300, 0.0, 1e-300, 1.0, 1e300], size=n)
    y = y * rng.uniform(1, 2, size=n)
    assert ot1d(x, y) == pytest.approx(wasserstein_distance(x, y))
    mu = rng.uniform(size=m)
    mu /= mu.sum()
    assert ot1d(y, x, None, mu) == pytest.approx(wasserstein_distance(y, x, None, mu))
