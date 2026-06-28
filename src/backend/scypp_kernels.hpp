#pragma once
// Portable CPU reference kernels for the ScyPP acceleration slots (CSR SpMV,
// pairwise squared-euclidean distance, separable 1-D correlation). Shared by the
// dispatcher's CPU fallback (backend.cpp) and the CPU-reference device backend
// (refgpu_backend.cpp), so the device path is byte-identical to the CPU path.

#include <cstdint>

namespace numpp {
namespace scypp_cpu {

// y[i] = sum_{k=indptr[i]..indptr[i+1]} data[k] * x[indices[k]]
template <class T>
void csr_spmv(int64_t rows, const int64_t* indptr, const int64_t* indices,
              const T* data, const T* x, T* y) {
  for (int64_t i = 0; i < rows; ++i) {
    T acc = T(0);
    for (int64_t k = indptr[i]; k < indptr[i + 1]; ++k) acc += data[k] * x[indices[k]];
    y[i] = acc;
  }
}

// D[i,j] = sum_d (A[i,d] - B[j,d])^2
template <class T>
void pairwise_sqdist(int64_t m, int64_t n, int64_t dim, const T* A, const T* B, T* D) {
  for (int64_t i = 0; i < m; ++i) {
    const T* a = A + i * dim;
    for (int64_t j = 0; j < n; ++j) {
      const T* b = B + j * dim;
      T s = T(0);
      for (int64_t d = 0; d < dim; ++d) {
        const T diff = a[d] - b[d];
        s += diff * diff;
      }
      D[i * n + j] = s;
    }
  }
}

// Resolve an out-of-range index to a valid source index per scipy.ndimage modes
// (0=reflect, 1=constant, 2=nearest, 3=mirror, 4=wrap). Returns -1 for constant
// (the caller substitutes cval).
inline int64_t bound_index(int64_t i, int64_t len, int mode) {
  if (i >= 0 && i < len) return i;
  if (len == 1) return mode == 1 ? -1 : 0;
  switch (mode) {
    case 1: return -1;                                          // constant -> cval
    case 2: return i < 0 ? 0 : len - 1;                         // nearest
    case 4: { i %= len; if (i < 0) i += len; return i; }        // wrap
    case 3: {                                                   // mirror: period 2n-2
      const int64_t p = 2 * len - 2;
      i %= p; if (i < 0) i += p;
      return i >= len ? p - i : i;
    }
    default: {                                                  // reflect: period 2n
      const int64_t p = 2 * len;
      i %= p; if (i < 0) i += p;
      return i >= len ? p - 1 - i : i;
    }
  }
}

// out[r,i] = sum_k weights[k] * in[r, i + k - (klen/2 + origin)]  (per row of len)
template <class T>
void separable_corr1d(int64_t lines, int64_t len, const T* in, const T* weights,
                      int64_t klen, int64_t origin, int mode, double cval, T* out) {
  const int64_t anchor = klen / 2 + origin;
  for (int64_t r = 0; r < lines; ++r) {
    const T* row = in + r * len;
    T* orow = out + r * len;
    for (int64_t i = 0; i < len; ++i) {
      T acc = T(0);
      for (int64_t k = 0; k < klen; ++k) {
        const int64_t bi = bound_index(i + k - anchor, len, mode);
        acc += weights[k] * (bi < 0 ? T(cval) : row[bi]);
      }
      orow[i] = acc;
    }
  }
}

}  // namespace scypp_cpu
}  // namespace numpp
