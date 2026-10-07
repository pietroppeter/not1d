# not1d

A minimal, partial port to Nim of [ot1d](https://github.com/stegua/ot1d) by Stefano Gualandi:
the Kantorovich-Wasserstein distance between two discrete measures on the real line. The `n`
stands for Nim, and the goal is not to replace ot1d but to show that
[nimlang-pypi](https://github.com/pietroppeter/uv-add-nimlang) (PyPI package `nimlang`) makes a
non-trivial Nim algorithm easy to ship to Python users. It also tests the usage of cpp backend with nimlang.

> AI disclosure: this project is mostly vibed. Currently [level 7](https://www.visidata.org/blog/2026/ai/#level-6%3A-bots-coded%2C-human-understands-mostly) on visidata AI scale: Human specced, bots coded.

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

[ot1d (C++) and not1d (Nim), side by side](docs/ot1d-vs-not1d.md) walks through the algorithm
on a small example and compares the two codebases, explaining the C++ for readers who know Nim
or Python.

The sort is the expensive part. Nim's `std/algorithm.sort` is a merge sort through a comparison
proc and was about 5x slower than ot1d overall; [`src/not1d/sorting.nim`](src/not1d/sorting.nim)
is a 60-line radix sort on the float bits that brings it on par with ot1d's pdqsort.

ot1d's own C++ pdqsort is available too, as a build-time switch, to compare the two:

```sh
NOT1D_SORT=pdqsort uv sync              # or uv build, uv run benchmark.py
uv run python -c "from not1d.core import sortAlgorithm; print(sortAlgorithm())"  # pdqsort
```

Nim calls `pdqsort.h` directly (`importcpp` in `sorting.nim`), which needs Nim's C++ backend:
[`src/not1d/core.nims`](src/not1d/core.nims) switches to it and compiles the C++ with the same
`zig cc` that nimlang uses for C, linking zig's libc++ statically. So the module needs no C++
runtime on the user's machine, and the pdqsort build cross-compiles like the default one (checked
by hand for nimlang's five wheel platforms).

With pdqsort, not1d runs as fast as ot1d on one thread: same algorithm, same sort, so the Nim
code adds no overhead. The radix sort is faster, most of all with weights, where pdqsort moves
(point, mass) pairs through a comparison function. On 1M points (ms):

| machine | case | not1d radix | not1d pdqsort | ot1d (1 thread) |
|:--------|:-----|------------:|--------------:|----------------:|
| Apple Silicon Mac | uniform | 16 | 32 | 31 |
| Apple Silicon Mac | weighted | 27 | 97 | 89 |
| 4-core Linux container | uniform | 60 | 60 | 55 |
| 4-core Linux container | weighted | 120 | 156 | 147 |

## How it is built

The whole package is `pyproject.toml`, two Nim files and a thin Python wrapper:

```
pyproject.toml
nimlang.lock                 # pinned commits of the Nim dependencies
src/not1d/__init__.py        # ot1d(): argument checks, lists to numpy arrays
src/not1d/core.nim           # the algorithm, importable as not1d.core
src/not1d/sorting.nim        # radix sort, or ot1d's pdqsort with NOT1D_SORT=pdqsort
src/not1d/core.nims          # build options: the C++ backend for pdqsort
src/not1d/pdqsort.h          # from ot1d (zlib license), and pdqsort_wrap.h
docs/ot1d-vs-not1d.md        # explainer: ot1d's C++ and not1d's Nim side by side
```

```toml
[build-system]
requires = ["hatchling", "nimlang>=0.0.2"]
build-backend = "hatchling.build"

[tool.hatch.build.hooks.nimlang]
extensions = ["src/not1d/core.nim"]

[tool.nimlang]
dependencies = ["nimpy", "nimpy_numpy >= 0.1.0"]
```

[nimpy-numpy](https://github.com/pietroppeter/nimpy-numpy) lets the Nim procs take numpy arrays
directly (`NumpyArray[float64]`, a view through the buffer protocol).

not1d is not published on PyPI. Install it from GitHub; uv builds it with nimlang, so no Nim and
no C compiler are needed:

```sh
uv add git+https://github.com/pietroppeter/not1d
```

`uv build` would also give a wheel, and nimlang can cross-build wheels for other platforms
(see [uv-add-nimlang-lib-demo](https://github.com/pietroppeter/uv-add-nimlang-lib-demo)).

```sh
uv sync                      # builds the extension
uv run pytest tests          # checks against scipy.stats.wasserstein_distance
uv run benchmark.py          # not1d vs scipy and ot1d (a uv script: deps in its header)
```

## Benchmark

`uv run benchmark.py`, with `y` half the size of `x` (milliseconds, best of 5). Timings depend
on the machine; two runs:

On an Apple Silicon Mac:

| n | case | not1d | scipy | ot1d (1 thread) | ot1d (8 threads) |
|--:|:-----|------:|------:|----------------:|-----------------:|
| 1,000 | uniform | 0.04 | 0.06 | 0.01 | 0.37 |
| 1,000 | weighted | 0.03 | 0.07 | 0.01 | 0.40 |
| 100,000 | uniform | 1.56 | 19.37 | 2.54 | 1.58 |
| 100,000 | weighted | 2.55 | 20.01 | 7.44 | 2.80 |
| 1,000,000 | uniform | 16.21 | 244.98 | 28.94 | 12.89 |
| 1,000,000 | weighted | 26.87 | 249.92 | 89.70 | 25.69 |

On a 4-core Linux container:

| n | case | not1d | scipy | ot1d (1 thread) | ot1d (8 threads) |
|--:|:-----|------:|------:|----------------:|-----------------:|
| 1,000 | uniform | 0.04 | 0.11 | 0.02 | 2.67 |
| 1,000 | weighted | 0.06 | 0.15 | 0.02 | 2.77 |
| 100,000 | uniform | 2.88 | 31.30 | 4.53 | 6.85 |
| 100,000 | weighted | 7.68 | 31.65 | 11.56 | 9.84 |
| 1,000,000 | uniform | 48.59 | 350.76 | 53.14 | 41.71 |
| 1,000,000 | weighted | 109.63 | 400.70 | 144.35 | 89.30 |

ot1d is published on PyPI as an sdist only, whose build needs Cython and numpy without
declaring them: `benchmark.py` adds them through uv's `extra-build-dependencies`.

## License

[EUPL-1.2](LICENSE), the license of ot1d, since this is a port of its code.
