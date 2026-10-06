## Optimal transport in 1D: a minimal Nim port of the distance computation of
## ot1d (https://github.com/stegua/ot1d) by Stefano Gualandi.
##
## The Kantorovich-Wasserstein distance of order p between two discrete
## measures on the real line is computed by sorting the support points and
## walking both measures from left to right, moving mass greedily: this is
## the solution that the complementary slackness conditions of the
## transportation linear program give in 1D. O(n log n), the sort dominates.

import std/math
import nimpy, nimpy_numpy
import sorting

type
  Uniform = object
    ## n points of mass 1/n each, without allocating the weights.
    mass: float

func `[]`(u: Uniform, i: int): float {.inline.} = u.mass

func cost(d: float, p: int): float {.inline.} =
  if p == 1: abs(d) else: d * d

func transport[A, B](x: openArray[float], a: A, y: openArray[float], b: B,
                     p: int): float =
  ## Cost of the optimal plan between sorted supports `x`, `y` with masses
  ## `a`, `b` (same total mass): the mass of the leftmost point of one measure
  ## goes to the leftmost points of the other, until one is used up.
  let m = x.len
  let n = y.len
  var i, j = 0
  var ai = a[0]
  var bj = b[0]
  while i < m and j < n:
    let c = cost(x[i] - y[j], p)
    if ai == bj:
      result += ai * c
      inc i
      inc j
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

func finish(z: float, p: int): float =
  if p == 1: z else: sqrt(z)

proc check(p: int; x, y: NumpyArray[float64]) =
  if p notin [1, 2]:
    raise newException(ValueError, "p must be 1 or 2, got " & $p)
  if x.ndim != 1 or y.ndim != 1:
    raise newException(ValueError, "x and y must be 1D arrays")
  if x.len == 0 or y.len == 0:
    raise newException(ValueError, "x and y must not be empty")

proc toSeq(a: NumpyArray[float64]): seq[float] =
  ## A copy of a 1D array: the input is sorted, and numpy's memory must not be.
  result = newSeq[float](a.len)
  for i in 0 ..< a.len:
    result[i] = a[i]

proc ot1d(x, y: NumpyArray[float64], p: int = 1,
          sorting: bool = true): float {.exportpy.} =
  ## Wasserstein distance of order `p` (1 or 2) between the empirical
  ## measures with support points `x` and `y`, every point with the same mass.
  check(p, x, y)
  var x = x.toSeq
  var y = y.toSeq
  if sorting:
    x.sortFloats()
    y.sortFloats()
  if x.len == y.len:
    # Same size: the i-th smallest x goes entirely to the i-th smallest y.
    var z = 0.0
    for i in 0 ..< x.len:
      z += cost(x[i] - y[i], p)
    finish(z / x.len.float, p)
  else:
    finish(transport(x, Uniform(mass: 1 / x.len), y,
                     Uniform(mass: 1 / y.len), p), p)

proc sortByPoint(x, w: seq[float]): (seq[float], seq[float]) =
  var pairs = newSeq[(float, float)](x.len)
  for i in 0 ..< x.len:
    pairs[i] = (x[i], w[i])
  pairs.sortFloats()
  result = (newSeq[float](x.len), newSeq[float](x.len))
  for i, (xi, wi) in pairs:
    result[0][i] = xi
    result[1][i] = wi

proc ot1dWeighted(x, y, mu, nu: NumpyArray[float64], p: int = 1,
                  sorting: bool = true): float {.exportpy.} =
  ## Wasserstein distance of order `p` (1 or 2) between the measures with
  ## support points `x`, `y` and masses `mu`, `nu`. The masses must have the
  ## same total, as in ot1d (no normalization is done).
  check(p, x, y)
  if mu.ndim != 1 or nu.ndim != 1 or mu.len != x.len or nu.len != y.len:
    raise newException(ValueError, "mu and nu must have the length of x and y")
  var (x, mu, y, nu) = (x.toSeq, mu.toSeq, y.toSeq, nu.toSeq)
  if sorting:
    (x, mu) = sortByPoint(x, mu)
    (y, nu) = sortByPoint(y, nu)
  finish(transport(x, mu, y, nu, p), p)

proc sortAlgorithm(): string {.exportpy.} =
  ## Which sort this build uses: "radix" (Nim) or "pdqsort" (C++).
  sorting.sortAlgorithm
