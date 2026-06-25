#pragma once

#include <cstdint>

#include "numpp/core/dtype.hpp"

namespace numpp {

// Weak-linked optional LAPACK backend, same nullable-vtable model as BLAS. When
// no backend is compiled in, lapack_vtable() returns nullptr and the linalg
// routines use their portable pure-C++ kernels. All buffers are row-major in the
// given dtype (float32/float64/complex64/complex128); functions return false if
// the dtype/case is unsupported so the caller falls back.
struct LapackVTable {
  // Solve A X = B in place (A: n x n, B: n x nrhs). B holds the solution on return.
  bool (*gesv)(int n, int nrhs, DTypeId dt, void* A, void* B);
  // Inverse of A (n x n) into Ainv.
  bool (*inv)(int n, DTypeId dt, const void* A, void* Ainv);
  // Cholesky: lower L (n x n) of SPD/HPD A.
  bool (*potrf)(int n, DTypeId dt, const void* A, void* L);
  // SVD: A (m x n) -> U, S (real), Vh. full=full_matrices.
  bool (*gesdd)(int m, int n, DTypeId dt, const void* A, void* U, void* S, void* Vh, bool full);
  // General eigen: A (n x n) -> W (complex eigenvalues), V (eigenvectors, may be null).
  bool (*geev)(int n, DTypeId dt, const void* A, void* W, void* V);
  // Hermitian/symmetric eigen: A (n x n) -> W (real, ascending), V (eigenvectors).
  bool (*heevd)(int n, DTypeId dt, const void* A, void* W, void* V);
};

// Null unless a LAPACK backend translation unit provides a real one.
const LapackVTable* lapack_vtable();

}  // namespace numpp
