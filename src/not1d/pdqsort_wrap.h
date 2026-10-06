// Entry points for not1d/sorting.nim: pdqsort on doubles, and on Nim
// (point, mass) tuples by point. Nim's anonymous tuple fields are Field0, Field1.
#pragma once
#include <cstddef>
#include "pdqsort.h"

inline void not1d_pdqsort(double* a, std::size_t n) {
    pdqsort(a, a + n);
}

template <class T>
inline void not1d_pdqsort_pairs(T* a, std::size_t n) {
    pdqsort(a, a + n, [](const T& u, const T& v) { return u.Field0 < v.Field0; });
}
