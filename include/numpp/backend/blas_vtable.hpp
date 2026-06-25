#pragma once

#include <cstdint>

#include "numpp/core/dtype.hpp"

namespace numpp {

// C-ABI-ish vtable an optional BLAS backend registers. The core references it
// through a nullable pointer (weak-linked model): when no backend is compiled
// in, blas_vtable() returns nullptr and dispatch falls back to the CPU kernel.
struct BlasVTable {
  // Row-major C[m,n] = A[m,k] * B[k,n]. Returns false if the dtype is unsupported.
  bool (*gemm)(DTypeId dt, int64_t m, int64_t n, int64_t k,
               const void* A, const void* B, void* C);
};

// Null unless a backend translation unit provides a real one.
const BlasVTable* blas_vtable();

}  // namespace numpp
