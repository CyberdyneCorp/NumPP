#include "numpp/backend/gpu_vtable.hpp"

#include <cuda_runtime.h>

#include <vector>

// Real CUDA GPU backend, compiled only when NUMPP_WITH_CUDA=ON. Same weak-vtable
// shape as the OpenCL/refgpu backends: float32/float64 element-wise arithmetic
// (add/sub/mul/div), negate, sqrt and sum/prod reductions run on the device;
// exp, complex and integer dtypes decline to the CPU kernel (arithmetic and sqrt
// are IEEE-exact so device results equal the CPU path bitwise).
//
// Built to PTX for a virtual arch (e.g. compute_70) and JIT-compiled by the
// driver to the device's native arch at runtime — so an older nvcc still targets
// a newer GPU (validated: nvcc 12.0 PTX JIT'd to sm_120 on an RTX 5060).

namespace numpp {
namespace {

__device__ inline float npp_sqrt(float x) { return sqrtf(x); }
__device__ inline double npp_sqrt(double x) { return sqrt(x); }

template <class T>
__global__ void k_binary(int op, const T* a, const T* b, T* o, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  T x = a[i], y = b[i], r;
  switch (op) { case kGAdd: r = x + y; break; case kGSub: r = x - y; break;
                case kGMul: r = x * y; break; default: r = x / y; }
  o[i] = r;
}
template <class T>
__global__ void k_unary(int op, const T* a, T* o, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  T x = a[i];
  o[i] = (op == kGSqrt) ? npp_sqrt(x) : -x;
}
template <class T>
__global__ void k_reduce(int op, const T* a, long n, T* part) {
  long gid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long gsz = (long)gridDim.x * blockDim.x;
  T acc = (op == kGProd) ? T(1) : T(0);
  for (long i = gid; i < n; i += gsz) acc = (op == kGProd) ? acc * a[i] : acc + a[i];
  part[gid] = acc;
}

template <class T>
__global__ void k_gemm(long M, long N, long K, const T* A, const T* B, T* C) {
  long i = blockIdx.y * (long)blockDim.y + threadIdx.y;
  long j = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= M || j >= N) return;
  T acc = T(0);
  for (long p = 0; p < K; ++p) acc += A[i * K + p] * B[p * N + j];
  C[i * N + j] = acc;
}

bool have_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <class T>
bool elementwise(int op, int64_t n, const void* a, const void* b, void* out, bool unary) {
  const size_t bytes = (size_t)n * sizeof(T);
  T *da = nullptr, *db = nullptr, *dout = nullptr;
  bool ok = cudaMalloc(&da, bytes) == cudaSuccess && cudaMalloc(&dout, bytes) == cudaSuccess &&
            (unary || cudaMalloc(&db, bytes) == cudaSuccess);
  if (ok) {
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    if (!unary) cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);
    const int threads = 256;
    const long blocks = (n + threads - 1) / threads;
    if (unary) k_unary<T><<<blocks, threads>>>(op, da, dout, n);
    else k_binary<T><<<blocks, threads>>>(op, da, db, dout, n);
    ok = cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  cudaFree(da); cudaFree(db); cudaFree(dout);
  return ok;
}

template <class T>
bool reduce_t(int op, int64_t n, const void* a, void* out) {
  const long P = 4096;  // partial accumulators
  T *da = nullptr, *dpart = nullptr;
  bool ok = cudaMalloc(&da, (size_t)n * sizeof(T)) == cudaSuccess &&
            cudaMalloc(&dpart, P * sizeof(T)) == cudaSuccess;
  std::vector<T> partial(P);
  if (ok) {
    cudaMemcpy(da, a, (size_t)n * sizeof(T), cudaMemcpyHostToDevice);
    const int threads = 256;
    k_reduce<T><<<P / threads, threads>>>(op, da, (long)n, dpart);
    ok = cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(partial.data(), dpart, P * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  cudaFree(da); cudaFree(dpart);
  if (!ok) return false;
  T acc = (op == kGProd) ? T(1) : T(0);
  for (long j = 0; j < P; ++j) acc = (op == kGProd) ? acc * partial[j] : acc + partial[j];
  *static_cast<T*>(out) = acc;
  return true;
}

template <class T>
bool gemm_t(int64_t m, int64_t n, int64_t k, const void* A, const void* B, void* C) {
  T *da = nullptr, *db = nullptr, *dc = nullptr;
  bool ok = cudaMalloc(&da, (size_t)m * k * sizeof(T)) == cudaSuccess &&
            cudaMalloc(&db, (size_t)k * n * sizeof(T)) == cudaSuccess &&
            cudaMalloc(&dc, (size_t)m * n * sizeof(T)) == cudaSuccess;
  if (ok) {
    cudaMemcpy(da, A, (size_t)m * k * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(db, B, (size_t)k * n * sizeof(T), cudaMemcpyHostToDevice);
    dim3 blk(16, 16);
    dim3 grid((unsigned)((n + 15) / 16), (unsigned)((m + 15) / 16));
    k_gemm<T><<<grid, blk>>>((long)m, (long)n, (long)k, da, db, dc);
    ok = cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(C, dc, (size_t)m * n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  cudaFree(da); cudaFree(db); cudaFree(dc);
  return ok;
}
bool gemm_impl(DTypeId dt, int64_t m, int64_t n, int64_t k, const void* A, const void* B, void* C) {
  if (!have_device()) return false;
  if (dt == DTypeId::Float32) return gemm_t<float>(m, n, k, A, B, C);
  if (dt == DTypeId::Float64) return gemm_t<double>(m, n, k, A, B, C);
  return false;
}

bool ew_binary(int op, DTypeId dt, int64_t n, const void* a, const void* b, void* out) {
  if (op < kGAdd || op > kGDiv || !have_device()) return false;
  if (dt == DTypeId::Float32) return elementwise<float>(op, n, a, b, out, false);
  if (dt == DTypeId::Float64) return elementwise<double>(op, n, a, b, out, false);
  return false;
}
bool ew_unary(int op, DTypeId dt, int64_t n, const void* a, void* out) {
  if ((op != kGNeg && op != kGSqrt) || !have_device()) return false;  // exp declined (ULP)
  if (dt == DTypeId::Float32) return elementwise<float>(op, n, a, nullptr, out, true);
  if (dt == DTypeId::Float64) return elementwise<double>(op, n, a, nullptr, out, true);
  return false;
}
bool reduce_impl(int op, DTypeId dt, int64_t n, const void* a, void* out) {
  if ((op != kGSum && op != kGProd) || !have_device()) return false;
  if (dt == DTypeId::Float32) return reduce_t<float>(op, n, a, out);
  if (dt == DTypeId::Float64) return reduce_t<double>(op, n, a, out);
  return false;
}

const GpuVTable g_vtable{"cuda", &ew_binary, &ew_unary, &reduce_impl, &gemm_impl};

}  // namespace

const GpuVTable* gpu_vtable() { return have_device() ? &g_vtable : nullptr; }

}  // namespace numpp
