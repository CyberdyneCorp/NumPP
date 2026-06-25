#include "numpp/backend/blas_vtable.hpp"

#include <complex>

#include <cblas.h>

// Optional BLAS backend. Compiled only when NUMPP_WITH_BLAS=ON (CMake links a
// CBLAS implementation). Provides a real vtable, registering it via blas_vtable().

namespace numpp {
namespace {

bool gemm_impl(DTypeId dt, int64_t m, int64_t n, int64_t k,
               const void* A, const void* B, void* C) {
  const int M = static_cast<int>(m), N = static_cast<int>(n), K = static_cast<int>(k);
  switch (dt) {
    case DTypeId::Float32: {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                  static_cast<const float*>(A), K, static_cast<const float*>(B), N, 0.0f,
                  static_cast<float*>(C), N);
      return true;
    }
    case DTypeId::Float64: {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
                  static_cast<const double*>(A), K, static_cast<const double*>(B), N, 0.0,
                  static_cast<double*>(C), N);
      return true;
    }
    case DTypeId::Complex64: {
      const std::complex<float> one(1, 0), zero(0, 0);
      cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &one, A, K, B, N, &zero, C, N);
      return true;
    }
    case DTypeId::Complex128: {
      const std::complex<double> one(1, 0), zero(0, 0);
      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &one, A, K, B, N, &zero, C, N);
      return true;
    }
    default:
      return false;  // dispatcher falls back to the CPU kernel
  }
}

const BlasVTable g_vtable{&gemm_impl};

}  // namespace

const BlasVTable* blas_vtable() { return &g_vtable; }

}  // namespace numpp
