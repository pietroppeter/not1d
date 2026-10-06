# not1d

A minimal, partial port to Nim of [ot1d](https://github.com/stegua/ot1d) by Stefano Gualandi:
the Kantorovich-Wasserstein distance between two discrete measures on the real line. The `n`
stands for Nim, and the goal is not to replace ot1d but to show that
[nimlang](https://github.com/pietroppeter/uv-add-nimlang) makes a non-trivial Nim algorithm
easy to ship to Python users.

```python
import numpy as np
from not1d import ot1d

x = np.random.uniform(1, 2, 1_000_000)
y = np.random.uniform(0, 1, 500_000)
ot1d(x, y)                  # W1, uniform masses
ot1d(x, y, p=2)             # W2
ot1d(x, y, mu, nu)          # with masses mu, nu (same total)
ot1d(x, y, sorting=False)   # x and y already sorted
```

## What is ported

| ot1d | not1d |
|:-----|:------|
| `OT1D(x, y, mu, nu, p=1/2, sorting)`, distance only | ported |
| `plan=True` (the optimal transport plan) | not ported |
| `threads` (parallel sort, parasort + pdqsort) | not ported: one thread, LSD radix sort |
| `parasort(x, mu)` | not ported |

The algorithm is ot1d's: sort the support points, then walk both measures from the left, moving
mass greedily (the solution that the complementary slackness conditions of the transportation
linear program give in 1D). It is in [`src/not1d/core.nim`](src/not1d/core.nim), about 100 lines
of Nim. Uniform masses are a tiny `Uniform` type with a `[]`, so one generic `transport` proc
serves both the uniform and the weighted case with no weights allocated.

The sort is the expensive part. Nim's `std/algorithm.sort` is a merge sort through a comparison
proc and was about 5x slower than ot1d overall; [`src/not1d/sorting.nim`](src/not1d/sorting.nim)
is a 60-line radix sort on the float bits that brings it on par with ot1d's pdqsort.

## How it is built

The whole package is `pyproject.toml`, two Nim files and a thin Python wrapper:

```
pyproject.toml
nimlang.lock                 # pinned commits of the Nim dependencies
src/not1d/__init__.py        # ot1d(): argument checks, lists to numpy arrays
src/not1d/core.nim           # the algorithm, importable as not1d.core
src/not1d/sorting.nim        # radix sort
```

```toml
[build-system]
requires = ["hatchling", "nimlang>=0.0.2"]
build-backend = "hatchling.build"

[tool.hatch.build.hooks.nimlang]
extensions = ["src/not1d/core.nim"]

[tool.nimlang]
dependencies = ["nimpy", "https://github.com/pietroppeter/nimpy-numpy"]
```

[nimpy-numpy](https://github.com/pietroppeter/nimpy-numpy) lets the Nim procs take numpy arrays
directly (`NumpyArray[float64]`, a view through the buffer protocol). The CI builds wheels for
Linux, macOS and Windows on one Linux machine and tests each one on its own OS, with no Nim and
no C compiler installed.

```sh
uv sync                      # builds the extension
uv run pytest tests          # checks against scipy.stats.wasserstein_distance
uv run python benchmark.py   # not1d vs scipy (and ot1d, if installed)
```

## Benchmark

`benchmark.py` on a 4-core Linux container, with `y` half the size of `x` (milliseconds, best of 5):

| n | case | not1d | scipy | ot1d (1 thread) | ot1d (8 threads) |
|--:|:-----|------:|------:|----------------:|-----------------:|
| 1,000 | uniform | 0.04 | 0.11 | 0.02 | 2.67 |
| 1,000 | weighted | 0.06 | 0.15 | 0.02 | 2.77 |
| 100,000 | uniform | 2.88 | 31.30 | 4.53 | 6.85 |
| 100,000 | weighted | 7.68 | 31.65 | 11.56 | 9.84 |
| 1,000,000 | uniform | 48.59 | 350.76 | 53.14 | 41.71 |
| 1,000,000 | weighted | 109.63 | 400.70 | 144.35 | 89.30 |

ot1d is published on PyPI as an sdist only, which needs Cython and numpy installed first:
`uv pip install numpy cython setuptools && uv pip install --no-build-isolation ot1d`.

## License

[EUPL-1.2](LICENSE), the license of ot1d, since this is a port of its code.
