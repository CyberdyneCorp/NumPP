#include "numpp/backend/gpu_vtable.hpp"

#include <cuda_runtime.h>

#include <mutex>
#include <vector>

// Real CUDA GPU backend, compiled only when NUMPP_WITH_CUDA=ON. Same weak-vtable
// shape as the OpenCL/refgpu backends: float32/float64 element-wise arithmetic
// (add/sub/mul/div), negate, sqrt, sum/prod reductions and GEMM run on the
// device; exp, complex and integer dtypes decline to the CPU kernel.
//
// Built to PTX for a virtual arch (compute_70) and JIT-compiled by the driver to
// the device's native arch at runtime — so an older nvcc still targets a newer
// GPU (validated: nvcc 12.0 PTX JIT'd to sm_120 on an RTX 5060).
//
// Device allocations are served from a small reuse pool so repeated ops don't pay
// cudaMalloc/cudaFree each call; GEMM uses a 16x16 shared-memory tiled kernel.

namespace numpp {
namespace {

constexpr int TILE = 16;

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
// 16x16 shared-memory tiled GEMM: C[m,n] = A[m,k] * B[k,n], row-major.
template <class T>
__global__ void k_gemm_tiled(long M, long N, long K, const T* A, const T* B, T* C) {
  __shared__ T As[TILE][TILE];
  __shared__ T Bs[TILE][TILE];
  const int ty = threadIdx.y, tx = threadIdx.x;
  const long row = blockIdx.y * (long)TILE + ty;
  const long col = blockIdx.x * (long)TILE + tx;
  T acc = T(0);
  const long ntiles = (K + TILE - 1) / TILE;
  for (long t = 0; t < ntiles; ++t) {
    const long ac = t * TILE + tx;
    const long br = t * TILE + ty;
    As[ty][tx] = (row < M && ac < K) ? A[row * K + ac] : T(0);
    Bs[ty][tx] = (br < K && col < N) ? B[br * N + col] : T(0);
    __syncthreads();
    for (int p = 0; p < TILE; ++p) acc += As[ty][p] * Bs[p][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

bool have_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// ---- device-buffer reuse pool (avoids cudaMalloc/cudaFree per call) ----
struct Buf { void* ptr = nullptr; size_t cap = 0; };
struct DevPool {
  std::mutex mu;
  std::vector<Buf> free_;
  Buf acquire(size_t bytes) {
    std::lock_guard<std::mutex> lk(mu);
    for (size_t i = 0; i < free_.size(); ++i) {
      if (free_[i].cap >= bytes) { Buf b = free_[i]; free_.erase(free_.begin() + i); return b; }
    }
    Buf b;
    if (cudaMalloc(&b.ptr, bytes) == cudaSuccess) b.cap = bytes;
    return b;  // ptr stays null on failure
  }
  void release(Buf b) {
    if (!b.ptr) return;
    std::lock_guard<std::mutex> lk(mu);
    if (free_.size() < 32) free_.push_back(b);
    else cudaFree(b.ptr);
  }
};
DevPool& pool() { static DevPool p; return p; }

template <class T>
bool elementwise(int op, int64_t n, const void* a, const void* b, void* out, bool unary) {
  const size_t bytes = (size_t)n * sizeof(T);
  Buf da = pool().acquire(bytes), dout = pool().acquire(bytes);
  Buf db = unary ? Buf{} : pool().acquire(bytes);
  bool ok = da.ptr && dout.ptr && (unary || db.ptr);
  if (ok) {
    cudaMemcpy(da.ptr, a, bytes, cudaMemcpyHostToDevice);
    if (!unary) cudaMemcpy(db.ptr, b, bytes, cudaMemcpyHostToDevice);
    const int threads = 256;
    const long blocks = (n + threads - 1) / threads;
    if (unary) k_unary<T><<<blocks, threads>>>(op, (const T*)da.ptr, (T*)dout.ptr, n);
    else k_binary<T><<<blocks, threads>>>(op, (const T*)da.ptr, (const T*)db.ptr, (T*)dout.ptr, n);
    ok = cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(out, dout.ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  pool().release(da); pool().release(db); pool().release(dout);
  return ok;
}

template <class T>
bool reduce_t(int op, int64_t n, const void* a, void* out) {
  const long P = 4096;
  Buf da = pool().acquire((size_t)n * sizeof(T)), dpart = pool().acquire(P * sizeof(T));
  std::vector<T> partial(P);
  bool ok = da.ptr && dpart.ptr;
  if (ok) {
    cudaMemcpy(da.ptr, a, (size_t)n * sizeof(T), cudaMemcpyHostToDevice);
    const int threads = 256;
    k_reduce<T><<<P / threads, threads>>>(op, (const T*)da.ptr, (long)n, (T*)dpart.ptr);
    ok = cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(partial.data(), dpart.ptr, P * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  pool().release(da); pool().release(dpart);
  if (!ok) return false;
  T acc = (op == kGProd) ? T(1) : T(0);
  for (long j = 0; j < P; ++j) acc = (op == kGProd) ? acc * partial[j] : acc + partial[j];
  *static_cast<T*>(out) = acc;
  return true;
}

template <class T>
bool gemm_t(int64_t m, int64_t n, int64_t k, const void* A, const void* B, void* C) {
  Buf da = pool().acquire((size_t)m * k * sizeof(T));
  Buf db = pool().acquire((size_t)k * n * sizeof(T));
  Buf dc = pool().acquire((size_t)m * n * sizeof(T));
  bool ok = da.ptr && db.ptr && dc.ptr;
  if (ok) {
    cudaMemcpy(da.ptr, A, (size_t)m * k * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(db.ptr, B, (size_t)k * n * sizeof(T), cudaMemcpyHostToDevice);
    dim3 blk(TILE, TILE);
    dim3 grid((unsigned)((n + TILE - 1) / TILE), (unsigned)((m + TILE - 1) / TILE));
    k_gemm_tiled<T><<<grid, blk>>>((long)m, (long)n, (long)k, (const T*)da.ptr, (const T*)db.ptr, (T*)dc.ptr);
    ok = cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(C, dc.ptr, (size_t)m * n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  pool().release(da); pool().release(db); pool().release(dc);
  return ok;
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
bool gemm_impl(DTypeId dt, int64_t m, int64_t n, int64_t k, const void* A, const void* B, void* C) {
  if (!have_device()) return false;
  if (dt == DTypeId::Float32) return gemm_t<float>(m, n, k, A, B, C);
  if (dt == DTypeId::Float64) return gemm_t<double>(m, n, k, A, B, C);
  return false;
}

const GpuVTable g_vtable{"cuda", &ew_binary, &ew_unary, &reduce_impl, &gemm_impl};

}  // namespace

const GpuVTable* gpu_vtable() { return have_device() ? &g_vtable : nullptr; }

}  // namespace numpp
