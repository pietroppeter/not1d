# ot1d (C++) and not1d (Nim), side by side

[ot1d](https://github.com/stegua/ot1d) computes the Kantorovich-Wasserstein distance between two
discrete measures on the real line, and in 1D that distance needs only a sort plus one linear
pass. This page walks through the algorithm, then reads ot1d's C++ and not1d's Nim side by side,
explaining the C++ as it comes. It is written for readers who know Nim (or Python) and little
C++.

The C++ excerpts are abridged from ot1d's
[`include/OT1D.hpp`](https://github.com/stegua/ot1d/blob/main/include/OT1D.hpp) and its Cython
files; the Nim excerpts are from [`src/not1d/core.nim`](../src/not1d/core.nim) and
[`src/not1d/sorting.nim`](../src/not1d/sorting.nim).

## 1. The problem: moving sand on a line

A discrete measure is a set of points with a mass on each. Think of `x = [0, 2]` as two piles of
sand of mass 1/2 each, and `y = [1, 3, 5]` as three holes of mass 1/3 each. The total mass is
the same (1) on both sides.

The Wasserstein distance of order p is the cheapest way to move all the sand into the holes,
where moving mass m over a distance d costs m × |d|^p. For p = 1 it is the classic Earth Mover's
Distance. For p = 2 the total cost is put under a square root at the end, so W2 is in the same
units as the points.

In general this is a linear program (the transportation problem) with m × n unknowns: how much
mass goes from each x[i] to each y[j]. In 1D there is a shortcut. An optimal plan never makes
two transports cross: if a pile at a goes to a hole at d and a pile at b goes to a hole at c,
with a < b and c < d, swapping the destinations is never more expensive. So, once both sides are
sorted, the leftmost sand goes to the leftmost holes, and so on. ot1d's README derives the same
thing from the complementary slackness conditions of the linear program.

The consequence: the whole algorithm is sort both sides, then walk left to right. The sort costs
O(n log n) and the walk O(m + n), so the sort dominates. That is why most of not1d's performance
work was about sorting.

## 2. The algorithm on a tiny example

With `x = [0, 2]` (mass 1/2 each) and `y = [1, 3, 5]` (mass 1/3 each), W1 = 2. The walk keeps two
cursors, `i` on x and `j` on y, and two remainders: `a`, the mass still left at x[i], and `b`,
the room still left at y[j]. At each step it moves min(a, b) from x[i] to y[j], then advances
whichever side ran out (both, if they ran out together).

| step | from x[i] | to y[j] | a (left at x[i]) | b (room at y[j]) | mass moved | cost = mass × distance | then |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 1 | 1/2 | 1/3 | 1/3 | 1/3 × 1 = 1/3 | y full: j+1, a = 1/6 |
| 2 | 0 | 3 | 1/6 | 1/3 | 1/6 | 1/6 × 3 = 1/2 | x empty: i+1, b = 1/6 |
| 3 | 2 | 3 | 1/2 | 1/6 | 1/6 | 1/6 × 1 = 1/6 | y full: j+1, a = 1/3 |
| 4 | 2 | 5 | 1/3 | 1/3 | 1/3 | 1/3 × 3 = 1 | both done |

Total cost 1/3 + 1/2 + 1/6 + 1 = 2. For p = 2 each cost uses the distance squared instead
(1/3 + 3/2 + 1/6 + 3 = 5), and the result is √5 ≈ 2.236.

The pairs (i, j) visited in the table are exactly the transport plan, which ot1d can return with
`plan=True` and not1d does not port. There are at most m + n − 1 of them, because every step
advances at least one cursor.

When both sides have the same number of points with equal masses, the walk degenerates: the k-th
smallest x goes entirely to the k-th smallest y. ot1d has a separate function for that case
(`OT1Da0`), and so does not1d (the `x.len == y.len` branch): the cost is just the mean of
|x[k] − y[k]|^p over k.

## 3. Side by side: the walk in C++ and in Nim

ot1d writes the walk eight times (p = 1 or 2, weighted or not, equal sizes or not, with or
without plan), while not1d writes it once as a generic proc. The logic is the same line for line;
the differences are in how memory and types are handled.

### ot1d: the unweighted walk (`OT1Da` in `include/OT1D.hpp`, abridged)

```cpp
double OT1Da(int m, int n, double* x0, double* y0, bool sorting, int threads=8) {
    int i = 0;
    int j = 0;
    double a = 1.0/m;
    double b = 1.0/n;
    double z = 0.0;

    // Copy input vector
    double* x = (double*)malloc(m*sizeof(double));
    double* y = (double*)malloc(n*sizeof(double));
    memcpy(x, x0, m*sizeof(double));
    memcpy(y, y0, n*sizeof(double));

    if (sorting) { pdqsort(x, x+m); pdqsort(y, y+n); }  // or parasort with threads

    while (i < m && j < n) {
        double d = fabs(x[i] - y[j]);
        if (a == b) {
            z += a*d;  a = 1.0/m;  b = 1.0/n;  i = i+1;  j = j+1;
        } else if (a > b) {
            z += b*d;  a = a - b;  b = 1.0/n;  j = j + 1;
        } else {
            z += a*d;  b = b - a;  a = 1.0/m;  i = i + 1;
        }
    }
    free(x);
    free(y);
    return z;
}
```

The C++ you need to read it:

- `double* x0` is a pointer: the address of the first double of an array that lives somewhere
  else (here, inside the numpy array). The function gets no length with it, so `m` and `n`
  travel as separate arguments. `x0[i]` reads the i-th element with no bounds check. Nim's
  equivalent is `ptr UncheckedArray[float]`; Nim code normally uses `openArray[float]`, which
  carries the length along.
- `int threads=8` is a default argument, like Nim's `threads = 8`.
- `malloc(m*sizeof(double))` asks for raw memory for m doubles, `memcpy` copies bytes into it,
  and `free` gives it back. Forget `free` and you leak; use x after `free` and you get garbage
  or a crash. The copy exists because sorting happens in place and must not reorder the
  caller's array. In Nim, `var x = toSeq(...)` makes the copy and the memory is released
  automatically when x goes out of scope.
- `(double*)malloc(...)` is a C-style cast: malloc returns an untyped pointer (`void*`) and the
  cast tells the compiler what it points to. Nim's `cast[ptr float](...)` is the same idea.
- `fabs` is the C absolute value for doubles; `z += a*d` is the same as in Nim.

### not1d: one generic walk (`transport` in `src/not1d/core.nim`)

```nim
type Uniform = object
  mass: float                      # n points of mass 1/n, no array needed

func `[]`(u: Uniform, i: int): float {.inline.} = u.mass

func cost(d: float, p: int): float {.inline.} =
  if p == 1: abs(d) else: d * d

func transport[A, B](x: openArray[float], a: A, y: openArray[float], b: B,
                     p: int): float =
  let m = x.len
  let n = y.len
  var i, j = 0
  var ai = a[0]
  var bj = b[0]
  while i < m and j < n:
    let c = cost(x[i] - y[j], p)
    if ai == bj:
      result += ai * c
      inc i; inc j
      if i < m: ai = a[i]
      if j < n: bj = b[j]
    elif ai > bj:
      result += bj * c
      ai -= bj
      inc j
      if j < n: bj = b[j]
    else:
      result += ai * c
      bj -= ai
      inc i
      if i < m: ai = a[i]
```

What changed, and why:

- One function instead of eight. `A` and `B` are generic: they can be a `seq[float]` of real
  weights or a `Uniform`, a tiny object whose `[]` always returns 1/n. The compiler makes a
  specialised copy for each combination, so the uniform case costs nothing extra and allocates
  no weight array. C++ templates (section 4) do the same thing; ot1d just does not use them for
  this part.
- Generics exist only at compile time. Python can only call the exported procs, which are not
  generic: `ot1d` calls `transport` with two `Uniform`s and `ot1dWeighted` with two
  `seq[float]`s (the sorted copies of the numpy masses), so the compiled module contains exactly
  those two versions. Python lists never reach Nim: the Python wrapper turns them into numpy
  arrays first, and also builds the uniform masses when only one side has weights.
- `p` is a parameter instead of separate functions. `cost` is `{.inline.}`, and `p` is the same
  for the whole loop, so the branch on p is well predicted and costs very little.
- `result` is Nim's implicit return variable, starting at 0.0. C++ has no such thing, hence
  ot1d's `double z = 0.0; ... return z;`.
- No `malloc` or `free`: `transport` reads arrays that the caller owns, and the copies are made
  by the caller as `seq`s, which Nim frees automatically.
- The `if i < m:` guards. ot1d's weighted version (`OT1Dc`) reads `mu[i]` right after `i = i+1`
  even when i has just reached m, one element past the end of the vector. It usually reads
  harmless memory and the loop then stops, but it is undefined behaviour in C++. In Nim, with
  bounds checks on (the default, also with `-d:release`), the same read would raise an error,
  so the guard is needed.

### The weighted case: `std::vector<std::pair<double,double>>`

With weights, ot1d keeps each point together with its mass, so that sorting points also moves
the masses:

```cpp
typedef std::pair<double, double> dbl_pair_t;     // a (point, mass) pair
std::vector<dbl_pair_t> mu;                       // a growable array of pairs
mu.reserve(m);                                    // allocate room for m up front
for (int i = 0; i < m; i++)
    mu.emplace_back(x[i], a[i]);                  // build a pair in place at the end
// ... sort mu by .first, then walk reading mu[i].first and mu[i].second
```

- `std::vector<T>` is C++'s `seq[T]`: it owns its memory and frees it when it goes out of
  scope, so no `free` is needed here.
- `std::pair<double, double>` is a two-field tuple with fields `.first` and `.second`; Nim's
  `(float, float)` with `[0]` and `[1]`.
- `typedef` gives a type a short name, like Nim's `type DblPair = (float, float)`.
- `reserve` + `emplace_back` is the C++ way to fill a vector without reallocating, like Nim's
  `newSeqOfCap` + `add`.

not1d does the same in `sortByPoint`: it zips x and the masses into a `seq[(float, float)]`,
sorts it by the first field, and unzips it back into two seqs for `transport`.

## 4. Sorting: pdqsort in C++, radix sort in Nim, and the bridge between them

Sorting is where the time goes, and it is where the two codebases differ most. ot1d uses
[pdqsort](https://github.com/orlp/pdqsort), a C++ template library; not1d uses a radix sort
written in Nim by default, and can call ot1d's pdqsort through Nim's C++ backend.

### Templates, iterators and lambdas: reading `pdqsort(x, x+m)`

```cpp
pdqsort(x, x + m);                                     // sort m doubles in place

pdqsort(mu.begin(), mu.end(),                          // sort pairs by point
        [](const dbl_pair_t& v, const dbl_pair_t& w) {
            return v.first < w.first;
        });
```

- `pdqsort` is a template:
  `template<class Iter, class Compare> void pdqsort(Iter begin, Iter end, Compare comp)`. Like
  a Nim generic `proc pdqsort[Iter, Compare](...)`, the compiler generates a separate copy of
  the code for each type it is used with, here once for `double*` and once for the pair vector.
- A template's code must be visible wherever it is used, so template libraries live entirely
  in headers. That is why pdqsort is just `pdqsort.h` with no `.cpp` next to it. Nim generics
  have the same property, which is why Nim generic code lives in the `.nim` files you import.
- C++ algorithms take a range as two iterators: a pointer to the first element and a pointer
  one past the last. For a plain array, `x` is the first and `x + m` is one past the end
  (pointer arithmetic moves by whole doubles, not bytes). Nim passes the array itself, as an
  `openArray`.
- `[](const dbl_pair_t& v, const dbl_pair_t& w) { return v.first < w.first; }` is a lambda, an
  anonymous function. `[]` lists the variables it captures from around it (none here), and
  `const T&` means "pass by reference, read-only", which avoids copying the pair. The Nim
  equivalent is `proc (v, w: (float, float)): bool = v[0] < w[0]`.
- With no comparator, pdqsort uses `<` on the elements; for doubles it then switches to a
  branchless partitioning that is especially fast on modern CPUs.
- `parasort(n, x, threads)` (the `threads` argument) splits the array across threads, sorts
  each chunk with pdqsort, and merges. not1d does not port it.

### not1d's default: an LSD radix sort (`src/not1d/sorting.nim`)

Nim's `std/algorithm.sort` is a merge sort that calls a comparison proc through a pointer, and
on 1M floats it made not1d about 5x slower than ot1d. Instead of comparing elements, a radix sort
distributes them by digits of their bit pattern:

1. Map each float64 to a uint64 with the same order: flip all bits of negatives, set the sign
   bit of positives (`sortableBits`). Now comparing the integers compares the floats.
2. Split the 64 bits into 6 digits of 11 bits (2048 buckets each). One pass over the data counts
   how many keys fall in each bucket, for all 6 digits at once.
3. For each digit, from the least significant up, scatter the elements into a temporary array
   at the bucket offsets (a prefix sum of the counts), then swap the roles of the two arrays. A
   digit where every key has the same value is skipped.
4. Arrays shorter than 64 use insertion sort instead.

That is 6 or 7 linear passes, no comparisons and no branches that depend on the data. The same
template (`radixSortImpl`) sorts plain floats and (point, mass) pairs, through a `key` template
that extracts the float to sort by. On an Apple Silicon Mac it is faster than pdqsort (16 vs 32
ms uniform, 27 vs 97 ms weighted, at 1M points), and on a 4-core Linux machine it is on par (see
the [README](../README.md) for the full tables).

### Calling pdqsort from Nim (`NOT1D_SORT=pdqsort`)

Nim compiles to C by default, or to C++ with the C++ backend. Only the C++ backend can call C++
templates, through `importcpp`:

```nim
proc pdqsort(a: ptr float, n: csize_t) {.importcpp: "not1d_pdqsort(@)",
                                         header: "pdqsort_wrap.h".}
```

```cpp
// pdqsort_wrap.h
inline void not1d_pdqsort(double* a, std::size_t n) {
    pdqsort(a, a + n);
}

template <class T>
inline void not1d_pdqsort_pairs(T* a, std::size_t n) {
    pdqsort(a, a + n, [](const T& u, const T& v) { return u.Field0 < v.Field0; });
}
```

- `importcpp: "not1d_pdqsort(@)"` tells Nim to emit that call literally into the generated
  C++, with `@` replaced by the arguments; `header:` makes it add `#include "pdqsort_wrap.h"`
  to the generated file.
- `inline` lets a function be defined in a header that several `.cpp` files include, without a
  "defined twice" link error.
- The pair version is itself a template over `T`, because the C++ name of Nim's tuple type
  `(float, float)` is generated by the Nim compiler. Nim's anonymous tuple fields are called
  `Field0` and `Field1` in the generated code, so the lambda compares `Field0`, the point.
- In Nim, the call is just `pdqsort(addr a[0], a.len.csize_t)`: the address of the first
  element and the count, i.e. the C++ iterator pair in another form.

With pdqsort, not1d runs at the same speed as ot1d on one thread (32 vs 31 ms uniform at 1M
points on an Apple Silicon Mac): same algorithm, same sort. The code generated from the Nim walk adds no
measurable overhead.

## 5. From Python to the compiled code

Both libraries hand numpy arrays to compiled code without copying them on the way in. ot1d does
it in three languages (C++, Cython, Python), not1d in two (Nim, Python).

### ot1d: C++ → Cython → Python

The C++ functions take raw pointers, which Python cannot pass. Cython, a Python-like language
that compiles to C or C++, is the glue. First `OT1D.pxd` declares the C++ functions so Cython
knows their signatures:

```cython
cdef extern from "OT1D.hpp":
    cdef double OT1Da(int, int, double*, double*, bool, int)
    cdef double OT1Dc(int, int, double*, double*, double*, double*, bool, int)
```

Then `OT1D.pyx` turns numpy arrays into pointers and picks the right C++ function:

```cython
cdef double OT_1Da(X, Y, sorting=True, threads=8):
    m = len(X)
    n = len(Y)
    cdef double[::1] Xmv = X          # typed memoryview: contiguous doubles
    cdef double[::1] Ymv = Y
    if m == n:
        return OT1Da0(n, &Xmv[0], &Ymv[0], sorting, threads)
    return OT1Da(m, n, &Xmv[0], &Ymv[0], sorting, threads)
```

- `double[::1]` is a typed memoryview: a view on a contiguous array of doubles, obtained
  through Python's buffer protocol, with no copy.
- `&Xmv[0]` takes the address of the first element, which is the `double*` the C++ function
  wants.
- The Python-level `OT1D(...)` function first calls `np.ascontiguousarray(X, dtype=float)` when
  the input is a list or not contiguous, then dispatches on `p`, on whether weights were given
  and on `plan` to one of the eight C++ functions.
- `setup.py` runs `cythonize`, which generates a large `.cpp` file from the `.pyx`, and
  setuptools compiles it with the system C++ compiler (flags like `-O2 -ffast-math`, and
  `-march=native` on Linux, which tunes the binary for the machine that builds it). It needs
  Cython and numpy at build time but does not declare them, which is why `pip install ot1d`
  fails unless they are installed first and build isolation is off (`--no-build-isolation`).

### not1d: Nim → Python

[nimpy](https://github.com/yglukhov/nimpy) exports a Nim proc to Python with one pragma, and
[nimpy-numpy](https://github.com/pietroppeter/nimpy-numpy) adds the numpy array type:

```nim
proc ot1d(x, y: NumpyArray[float64], p: int = 1,
          sorting: bool = true): float {.exportpy.} =
  check(p, x, y)
  var x = x.toSeq        # copy: we sort it
  var y = y.toSeq
  ...
```

- `{.exportpy.}` makes the proc callable from Python as `not1d.core.ot1d`; nimpy converts
  `int`, `bool` and `float` arguments and the return value.
- `NumpyArray[float64]` is nimpy-numpy's view on a numpy array through the same buffer protocol
  as Cython's memoryview, also with no copy. It checks the dtype and handles strided arrays
  (like `x[::2]`). The copy into a `seq` happens because the sort works in place, the same
  reason ot1d calls `memcpy`.
- The thin Python wrapper in [`src/not1d/__init__.py`](../src/not1d/__init__.py) converts lists
  with `np.asarray`, fills in uniform masses when only one side has weights, and checks
  arguments. Checks are in Python because nimpy turns a Nim `ValueError` into a
  `nimpy.ValueError`, which is not a Python `ValueError`.
- Building is the `[tool.hatch.build.hooks.nimlang]` table in `pyproject.toml`:
  [nimlang](https://github.com/pietroppeter/uv-add-nimlang) compiles `core.nim` to an extension
  module with the Nim and zig it ships, so `uv add git+https://github.com/pietroppeter/not1d`
  works on a machine with no Nim and no C compiler.

## 6. A bug in ot1d, and a C++ cheat sheet for Nim users

### The p = 2 parallel sort: a C++ trap in one character

In `OT1Db` (p = 2, unweighted, m ≠ n) the parallel branch sorts x with y's length:

```cpp
parasort(n, x, threads);   // should be parasort(m, x, threads)
parasort(n, y, threads);
```

The C++ compiler cannot catch this: `parasort` gets a bare pointer and a count, and believes the
count. If x is longer than y, only its first n elements are sorted and the distance is wrong
(1.29 instead of 0.117 in a test with 1000 and 333 points). If x is shorter, parasort writes
past the end of the malloc'd block and the process aborts with `free(): invalid size`. It only
happens with `threads > 1`, and the default is 8. The `threads == 1` branch uses
`pdqsort(x, x+m)` correctly.

This is the kind of bug Nim's `openArray` prevents: the length travels with the data, so there
is no separate count to get wrong. not1d's tests compare against ot1d with `threads=1` for this
reason.

### Other differences

- Exact float equality. Both walks test `a == b` to decide whether both sides ran out together,
  as ot1d does. With real-valued masses, two remainders that should be equal can differ by a
  rounding error; the walk then does one extra step that moves a tiny amount, and the result is
  unaffected. not1d keeps ot1d's logic so the two stay comparable.
- Mass totals are not checked or normalised, in either library: mu and nu must have the same
  total.
- `-ffast-math` in ot1d's build lets the C++ compiler reorder floating-point sums. Results can
  differ from not1d in the last digits, which is why the comparison tests use a relative
  tolerance of 1e-9.

### C++ to Nim, side by side

| C++ | Nim | Notes |
| --- | --- | --- |
| `double* x` | `ptr UncheckedArray[float]`, or `openArray[float]` | A pointer has no length; openArray carries it |
| `x[i]` on a pointer | `x[i]` | C++ never checks bounds; Nim checks unless `-d:danger` |
| `malloc` / `free` | `newSeq` / automatic | Nim's memory management frees seqs at scope end |
| `memcpy(dst, src, n*sizeof(double))` | `var x = x.toSeq` (or `copyMem`) | |
| `std::vector<double>` | `seq[float]` | Both own and free their memory |
| `std::pair<double,double>`, `.first` / `.second` | `(float, float)`, `[0]` / `[1]` | Generated C++ calls them `Field0` / `Field1` |
| `typedef T Name;` | `type Name = T` | |
| `template<class T> void f(T x)` | `proc f[T](x: T)` | Both generate one copy per type used |
| `[](const T& a, const T& b) { return a < b; }` | `proc (a, b: T): bool = a < b` | `const T&`: read-only reference, no copy |
| `f(x, x + n)` (iterator range) | `f(x)` with an `openArray` | `x + n` = one past the last element |
| `inline` in a header | `{.inline.}` | |
| `#include "file.h"` | `import file` | C++ pastes the file's text in; Nim imports a module |
| `.h` / `.hpp` | `.nim` | The extension is a convention; the compiler does not care |
| `int threads=8` | `threads = 8` | Default argument |
| Calling C++ from Nim | `{.importcpp: "f(@)", header: "f.h".}` | Needs Nim's C++ backend |
