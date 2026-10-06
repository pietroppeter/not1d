## Sorting floats by radix sort (LSD, 11 bits per pass). Nim's
## std/algorithm.sort is a merge sort through a comparison proc: on 1M floats
## it takes about 4x longer than the pdqsort used by ot1d; radix sort is
## faster than both.
##
## `key` maps an element to the float it is sorted by.

const
  bits = 11
  buckets = 1 shl bits
  passes = (64 + bits - 1) div bits

func sortableBits(f: float): uint64 {.inline.} =
  ## Maps a float to an unsigned int with the same order: flip all bits of
  ## negatives, only the sign bit of positives. (NaNs end up at the ends.)
  let u = cast[uint64](f)
  if (u shr 63) == 1: not u else: u or (1'u64 shl 63)

template radixSortImpl(a, key: untyped) =
  type T = typeof(a[0])
  let n = a.len
  if n < 64:
    # insertion sort
    for i in 1 ..< n:
      let v = a[i]
      var j = i - 1
      while j >= 0 and key(v) < key(a[j]):
        a[j + 1] = a[j]
        dec j
      a[j + 1] = v
  else:
    var counts: array[passes, array[buckets, int]]
    for v in a:
      let k = sortableBits(key(v))
      for p in 0 ..< passes:
        inc counts[p][(k shr (p * bits)) and (buckets - 1)]
    var tmp = newSeq[T](n)
    var inTmp = false
    for p in 0 ..< passes:
      if counts[p][(sortableBits(key(a[0])) shr (p * bits)) and (buckets - 1)] == n:
        continue  # every element has the same digit: nothing to do
      var offsets: array[buckets, int]
      var sum = 0
      for b in 0 ..< buckets:
        offsets[b] = sum
        sum += counts[p][b]
      template radixPass(src, dst: untyped) =
        for v in src:
          let d = (sortableBits(key(v)) shr (p * bits)) and (buckets - 1)
          dst[offsets[d]] = v
          inc offsets[d]
      if inTmp: radixPass(tmp, a) else: radixPass(a, tmp)
      inTmp = not inTmp
    if inTmp:
      for i in 0 ..< n: a[i] = tmp[i]

proc radixSort*(a: var openArray[float]) =
  template key(v: float): float = v
  radixSortImpl(a, key)

proc radixSort*(a: var openArray[(float, float)]) =
  ## Sorts (point, mass) pairs by point.
  template key(v: (float, float)): float = v[0]
  radixSortImpl(a, key)
