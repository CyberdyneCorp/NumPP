#pragma once

#include <cstdint>

#include "numpp/core/dtype.hpp"

namespace numpp {

// Op codes shared between the ufunc dispatcher and any GPU backend.
enum GpuBinOp { kGAdd = 0, kGSub, kGMul, kGDiv };
enum GpuUnOp { kGNeg = 0, kGSqrt, kGExp };
enum GpuRedOp { kGSum = 0, kGProd };

// Weak-linked optional GPU backend, same nullable-vtable model as BLAS/LAPACK.
// Buffers are contiguous, native-dtype; functions return false when the
// (op, dtype) pair is unsupported so the dispatcher falls back to the CPU kernel.
struct GpuVTable {
  const char* name;
  bool (*elementwise_binary)(int op, DTypeId dt, int64_t n, const void* a, const void* b, void* out);
  bool (*elementwise_unary)(int op, DTypeId dt, int64_t n, const void* a, void* out);
  bool (*reduce)(int op, DTypeId dt, int64_t n, const void* a, void* out);  // out: 1 element
  // Row-major C[m,n] = A[m,k] * B[k,n], accumulated in p-order — matches the CPU
  // gemm within floating-point tolerance (device FMA contraction makes it
  // accurate but not necessarily bitwise-identical). May be null if the backend
  // has no GEMM; returns false on an unsupported dtype so matmul falls back to CPU.
  bool (*gemm)(DTypeId dt, int64_t m, int64_t n, int64_t k, const void* A, const void* B, void* C);

  // ---- ScyPP acceleration slots (sparse / geometry / ndimage) ----
  // These host raw device kernels for operations whose high-level API lives in
  // ScyPP (scipy.*). Any slot may be null; the dispatcher falls back to NumPP's
  // portable CPU kernel, and a slot returning false also triggers the fallback.

  // CSR sparse matrix-vector product y = A·x. A is in CSR form: indptr (int64,
  // length rows+1), indices (int64, length nnz), data (dt, length nnz). x and y
  // are dt; y has length rows. dt is float32/float64.
  bool (*csr_spmv)(DTypeId dt, int64_t rows, int64_t cols, int64_t nnz, const int64_t* indptr,
                   const int64_t* indices, const void* data, const void* x, void* y);

  // Pairwise squared-euclidean distance matrix D[m,n] = sum_d (A[m,d]-B[n,d])^2,
  // A row-major (m × dim), B row-major (n × dim), D row-major (m × n). dt is
  // float32/float64. The euclidean square root, if wanted, is applied by caller.
  bool (*pairwise_sqdist)(DTypeId dt, int64_t m, int64_t n, int64_t dim,
                          const void* A, const void* B, void* D);

  // Separable 1-D correlation over `lines` contiguous rows of length `len`:
  //   out[r,i] = sum_k weights[k] * in[r, i + k - (klen/2 + origin)]
  // with the out-of-range source index resolved per `mode` (0=reflect, 1=constant
  // (cval), 2=nearest, 3=mirror, 4=wrap) — matching scipy.ndimage.correlate1d.
  // dt is float32/float64.
  bool (*separable_corr1d)(DTypeId dt, int64_t lines, int64_t len, const void* in,
                           const void* weights, int64_t klen, int64_t origin, int mode,
                           double cval, void* out);
};

// Null unless a GPU backend translation unit provides a real one.
const GpuVTable* gpu_vtable();

}  // namespace numpp
