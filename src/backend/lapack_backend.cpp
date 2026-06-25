#include "numpp/backend/lapack_vtable.hpp"

// Optional LAPACK backend. Compiled only when NUMPP_WITH_LAPACK=ON (CMake links
// a LAPACKE provider). Implements the subset the linalg routines route through;
// unsupported dtypes/cases return false so the portable fallback handles them.

#include <complex>
#include <vector>

#include <lapacke.h>

namespace numpp {
namespace {

bool gesv_impl(int n, int nrhs, DTypeId dt, void* A, void* B) {
  std::vector<lapack_int> ipiv(n);
  switch (dt) {
    case DTypeId::Float64:
      return LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, static_cast<double*>(A), n, ipiv.data(),
                           static_cast<double*>(B), nrhs) == 0;
    case DTypeId::Float32:
      return LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, static_cast<float*>(A), n, ipiv.data(),
                           static_cast<float*>(B), nrhs) == 0;
    case DTypeId::Complex128:
      return LAPACKE_zgesv(LAPACK_ROW_MAJOR, n, nrhs,
                           reinterpret_cast<lapack_complex_double*>(A), n, ipiv.data(),
                           reinterpret_cast<lapack_complex_double*>(B), nrhs) == 0;
    case DTypeId::Complex64:
      return LAPACKE_cgesv(LAPACK_ROW_MAJOR, n, nrhs,
                           reinterpret_cast<lapack_complex_float*>(A), n, ipiv.data(),
                           reinterpret_cast<lapack_complex_float*>(B), nrhs) == 0;
    default: return false;
  }
}

// Remaining entries (inv/potrf/gesdd/geev/heevd) can be filled with the matching
// LAPACKE_* calls; left null here so the portable kernels are used for them.
const LapackVTable g_vtable{&gesv_impl, nullptr, nullptr, nullptr, nullptr, nullptr};

}  // namespace

const LapackVTable* lapack_vtable() { return &g_vtable; }

}  // namespace numpp
