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

// ---- ScyPP acceleration kernels (CSR SpMV / pairwise sqdist / separable corr) ----
__device__ inline long long npp_bound_index(long long i, long long len, int mode) {
  if (i >= 0 && i < len) return i;
  if (len == 1) return mode == 1 ? -1 : 0;
  switch (mode) {
    case 1: return -1;                                    // constant -> cval
    case 2: return i < 0 ? 0 : len - 1;                   // nearest
    case 4: { i %= len; if (i < 0) i += len; return i; }  // wrap
    case 3: { long long p = 2 * len - 2; i %= p; if (i < 0) i += p; return i >= len ? p - i : i; }  // mirror
    default: { long long p = 2 * len; i %= p; if (i < 0) i += p; return i >= len ? p - 1 - i : i; } // reflect
  }
}
template <class T>
__global__ void k_csr_spmv(long rows, const long long* indptr, const long long* indices,
                           const T* data, const T* x, T* y) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= rows) return;
  T acc = T(0);
  for (long long k = indptr[i]; k < indptr[i + 1]; ++k) acc += data[k] * x[indices[k]];
  y[i] = acc;
}
template <class T>
__global__ void k_pairwise_sqdist(long m, long n, long dim, const T* A, const T* B, T* D) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= m * n) return;
  const T* a = A + (idx / n) * dim;
  const T* b = B + (idx % n) * dim;
  T s = T(0);
  for (long d = 0; d < dim; ++d) { T df = a[d] - b[d]; s += df * df; }
  D[idx] = s;
}
template <class T>
__global__ void k_corr1d(long lines, long len, const T* in, const T* w, long klen,
                         long anchor, int mode, double cval, T* out) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= lines * len) return;
  const long i = idx % len;
  const T* row = in + (idx / len) * len;
  T acc = T(0);
  for (long k = 0; k < klen; ++k) {
    long long bi = npp_bound_index((long long)(i + k - anchor), (long long)len, mode);
    acc += w[k] * (bi < 0 ? (T)cval : row[bi]);
  }
  out[idx] = acc;
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

template <class T>
bool csr_spmv_t(int64_t rows, int64_t cols, int64_t nnz, const int64_t* indptr,
                const int64_t* indices, const void* data, const void* x, void* y) {
  Buf dip = pool().acquire((size_t)(rows + 1) * sizeof(int64_t));
  Buf dix = pool().acquire((size_t)nnz * sizeof(int64_t));
  Buf dd = pool().acquire((size_t)nnz * sizeof(T));
  Buf dx = pool().acquire((size_t)cols * sizeof(T));
  Buf dy = pool().acquire((size_t)rows * sizeof(T));
  bool ok = dip.ptr && dix.ptr && dd.ptr && dx.ptr && dy.ptr;
  if (ok) {
    cudaMemcpy(dip.ptr, indptr, (size_t)(rows + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dix.ptr, indices, (size_t)nnz * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dd.ptr, data, (size_t)nnz * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dx.ptr, x, (size_t)cols * sizeof(T), cudaMemcpyHostToDevice);
    const int threads = 256;
    const long blocks = (rows + threads - 1) / threads;
    k_csr_spmv<T><<<blocks, threads>>>((long)rows, (const long long*)dip.ptr, (const long long*)dix.ptr,
                                       (const T*)dd.ptr, (const T*)dx.ptr, (T*)dy.ptr);
    ok = cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(y, dy.ptr, (size_t)rows * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  pool().release(dip); pool().release(dix); pool().release(dd); pool().release(dx); pool().release(dy);
  return ok;
}
template <class T>
bool pairwise_sqdist_t(int64_t m, int64_t n, int64_t dim, const void* A, const void* B, void* D) {
  Buf da = pool().acquire((size_t)m * dim * sizeof(T));
  Buf db = pool().acquire((size_t)n * dim * sizeof(T));
  Buf dd = pool().acquire((size_t)m * n * sizeof(T));
  bool ok = da.ptr && db.ptr && dd.ptr;
  if (ok) {
    cudaMemcpy(da.ptr, A, (size_t)m * dim * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(db.ptr, B, (size_t)n * dim * sizeof(T), cudaMemcpyHostToDevice);
    const int threads = 256;
    const long blocks = (m * n + threads - 1) / threads;
    k_pairwise_sqdist<T><<<blocks, threads>>>((long)m, (long)n, (long)dim, (const T*)da.ptr, (const T*)db.ptr, (T*)dd.ptr);
    ok = cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(D, dd.ptr, (size_t)m * n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  pool().release(da); pool().release(db); pool().release(dd);
  return ok;
}
template <class T>
bool corr1d_t(int64_t lines, int64_t len, const void* in, const void* w, int64_t klen,
              int64_t origin, int mode, double cval, void* out) {
  const size_t nbytes = (size_t)lines * len * sizeof(T);
  Buf din = pool().acquire(nbytes), dw = pool().acquire((size_t)klen * sizeof(T)), dout = pool().acquire(nbytes);
  bool ok = din.ptr && dw.ptr && dout.ptr;
  if (ok) {
    cudaMemcpy(din.ptr, in, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dw.ptr, w, (size_t)klen * sizeof(T), cudaMemcpyHostToDevice);
    const int threads = 256;
    const long blocks = (lines * len + threads - 1) / threads;
    k_corr1d<T><<<blocks, threads>>>((long)lines, (long)len, (const T*)din.ptr, (const T*)dw.ptr,
                                     (long)klen, (long)(klen / 2 + origin), mode, cval, (T*)dout.ptr);
    ok = cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(out, dout.ptr, nbytes, cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  pool().release(din); pool().release(dw); pool().release(dout);
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
bool csr_spmv_impl(DTypeId dt, int64_t rows, int64_t cols, int64_t nnz, const int64_t* indptr,
                   const int64_t* indices, const void* data, const void* x, void* y) {
  if (!have_device()) return false;
  if (dt == DTypeId::Float32) return csr_spmv_t<float>(rows, cols, nnz, indptr, indices, data, x, y);
  if (dt == DTypeId::Float64) return csr_spmv_t<double>(rows, cols, nnz, indptr, indices, data, x, y);
  return false;
}
bool pairwise_sqdist_impl(DTypeId dt, int64_t m, int64_t n, int64_t dim, const void* A, const void* B, void* D) {
  if (!have_device()) return false;
  if (dt == DTypeId::Float32) return pairwise_sqdist_t<float>(m, n, dim, A, B, D);
  if (dt == DTypeId::Float64) return pairwise_sqdist_t<double>(m, n, dim, A, B, D);
  return false;
}
bool separable_corr1d_impl(DTypeId dt, int64_t lines, int64_t len, const void* in, const void* weights,
                           int64_t klen, int64_t origin, int mode, double cval, void* out) {
  if (!have_device()) return false;
  if (dt == DTypeId::Float32) return corr1d_t<float>(lines, len, in, weights, klen, origin, mode, cval, out);
  if (dt == DTypeId::Float64) return corr1d_t<double>(lines, len, in, weights, klen, origin, mode, cval, out);
  return false;
}

const GpuVTable g_vtable{"cuda", &ew_binary, &ew_unary, &reduce_impl, &gemm_impl,
                         &csr_spmv_impl, &pairwise_sqdist_impl, &separable_corr1d_impl};

}  // namespace

const GpuVTable* gpu_vtable() { return have_device() ? &g_vtable : nullptr; }

}  // namespace numpp
