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
};

// Null unless a GPU backend translation unit provides a real one.
const GpuVTable* gpu_vtable();

}  // namespace numpp
