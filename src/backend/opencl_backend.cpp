#include "numpp/backend/gpu_vtable.hpp"

#include "opencl_min.h"

#include <mutex>
#include <string>
#include <vector>

// Real OpenCL GPU backend, compiled only when NUMPP_WITH_OPENCL=ON. Implements the
// GPU vtable with kernels that run on an OpenCL GPU device (e.g. NVIDIA via the ICD
// loader). float32/float64 only; everything else returns false so the ufunc
// dispatcher falls back to the portable CPU kernel.
//
// Arithmetic (add/sub/mul/div), negate and sqrt are IEEE-754 ops, bit-identical to
// the CPU path; exp (transcendental) and complex dtypes decline to the CPU kernel.
// Device allocations come from a small reuse pool; GEMM is a 16x16 local-memory
// tiled kernel.

namespace numpp {
namespace {

const char* kProgramSrc = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define TILE 16
__kernel void binop_f(int op, __global const float* a, __global const float* b, __global float* o) {
  int i = get_global_id(0); float x = a[i], y = b[i], r;
  switch (op) { case 0: r = x + y; break; case 1: r = x - y; break; case 2: r = x * y; break; default: r = x / y; }
  o[i] = r;
}
__kernel void binop_d(int op, __global const double* a, __global const double* b, __global double* o) {
  int i = get_global_id(0); double x = a[i], y = b[i], r;
  switch (op) { case 0: r = x + y; break; case 1: r = x - y; break; case 2: r = x * y; break; default: r = x / y; }
  o[i] = r;
}
__kernel void unop_f(int op, __global const float* a, __global float* o) {
  int i = get_global_id(0); float x = a[i]; o[i] = (op == 1) ? sqrt(x) : -x;
}
__kernel void unop_d(int op, __global const double* a, __global double* o) {
  int i = get_global_id(0); double x = a[i]; o[i] = (op == 1) ? sqrt(x) : -x;
}
__kernel void reduce_f(int op, __global const float* a, long n, __global float* part) {
  int gid = get_global_id(0); int gsz = get_global_size(0);
  float acc = (op == 1) ? 1.0f : 0.0f;
  for (long i = gid; i < n; i += gsz) acc = (op == 1) ? acc * a[i] : acc + a[i];
  part[gid] = acc;
}
__kernel void reduce_d(int op, __global const double* a, long n, __global double* part) {
  int gid = get_global_id(0); int gsz = get_global_size(0);
  double acc = (op == 1) ? 1.0 : 0.0;
  for (long i = gid; i < n; i += gsz) acc = (op == 1) ? acc * a[i] : acc + a[i];
  part[gid] = acc;
}
__kernel void gemm_f(long M, long N, long K, __global const float* A, __global const float* B, __global float* C) {
  __local float As[TILE][TILE]; __local float Bs[TILE][TILE];
  int tx = get_local_id(0), ty = get_local_id(1);
  long col = get_global_id(0), row = get_global_id(1);
  float acc = 0.0f; long nt = (K + TILE - 1) / TILE;
  for (long t = 0; t < nt; ++t) {
    long ac = t*TILE + tx, br = t*TILE + ty;
    As[ty][tx] = (row < M && ac < K) ? A[row*K + ac] : 0.0f;
    Bs[ty][tx] = (br < K && col < N) ? B[br*N + col] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = 0; p < TILE; ++p) acc += As[ty][p] * Bs[p][tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < M && col < N) C[row*N + col] = acc;
}
__kernel void gemm_d(long M, long N, long K, __global const double* A, __global const double* B, __global double* C) {
  __local double As[TILE][TILE]; __local double Bs[TILE][TILE];
  int tx = get_local_id(0), ty = get_local_id(1);
  long col = get_global_id(0), row = get_global_id(1);
  double acc = 0.0; long nt = (K + TILE - 1) / TILE;
  for (long t = 0; t < nt; ++t) {
    long ac = t*TILE + tx, br = t*TILE + ty;
    As[ty][tx] = (row < M && ac < K) ? A[row*K + ac] : 0.0;
    Bs[ty][tx] = (br < K && col < N) ? B[br*N + col] : 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = 0; p < TILE; ++p) acc += As[ty][p] * Bs[p][tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < M && col < N) C[row*N + col] = acc;
}
// ---- ScyPP acceleration kernels ----
long npp_bidx(long i, long len, int mode) {
  if (i >= 0 && i < len) return i;
  if (len == 1) return (mode == 1) ? -1 : 0;
  if (mode == 1) return -1;
  if (mode == 2) return (i < 0) ? 0 : len - 1;
  if (mode == 4) { i %= len; if (i < 0) i += len; return i; }
  if (mode == 3) { long p = 2*len - 2; i %= p; if (i < 0) i += p; return (i >= len) ? p - i : i; }
  long p = 2*len; i %= p; if (i < 0) i += p; return (i >= len) ? p - 1 - i : i;
}
__kernel void spmv_f(long rows, __global const long* indptr, __global const long* indices, __global const float* data, __global const float* x, __global float* y) {
  long i = get_global_id(0); if (i >= rows) return;
  float acc = 0.0f; for (long k = indptr[i]; k < indptr[i+1]; ++k) acc += data[k]*x[indices[k]]; y[i] = acc;
}
__kernel void spmv_d(long rows, __global const long* indptr, __global const long* indices, __global const double* data, __global const double* x, __global double* y) {
  long i = get_global_id(0); if (i >= rows) return;
  double acc = 0.0; for (long k = indptr[i]; k < indptr[i+1]; ++k) acc += data[k]*x[indices[k]]; y[i] = acc;
}
__kernel void pdist_f(long m, long n, long dim, __global const float* A, __global const float* B, __global float* D) {
  long idx = get_global_id(0); if (idx >= m*n) return;
  __global const float* a = A + (idx/n)*dim; __global const float* b = B + (idx%n)*dim;
  float s = 0.0f; for (long d = 0; d < dim; ++d) { float df = a[d]-b[d]; s += df*df; } D[idx] = s;
}
__kernel void pdist_d(long m, long n, long dim, __global const double* A, __global const double* B, __global double* D) {
  long idx = get_global_id(0); if (idx >= m*n) return;
  __global const double* a = A + (idx/n)*dim; __global const double* b = B + (idx%n)*dim;
  double s = 0.0; for (long d = 0; d < dim; ++d) { double df = a[d]-b[d]; s += df*df; } D[idx] = s;
}
__kernel void corr1d_f(long lines, long len, __global const float* in, __global const float* w, long klen, long anchor, int mode, float cval, __global float* out) {
  long idx = get_global_id(0); if (idx >= lines*len) return;
  long i = idx % len; __global const float* row = in + (idx/len)*len;
  float acc = 0.0f; for (long k = 0; k < klen; ++k) { long bi = npp_bidx(i+k-anchor, len, mode); acc += w[k]*((bi < 0) ? cval : row[bi]); }
  out[idx] = acc;
}
__kernel void corr1d_d(long lines, long len, __global const double* in, __global const double* w, long klen, long anchor, int mode, double cval, __global double* out) {
  long idx = get_global_id(0); if (idx >= lines*len) return;
  long i = idx % len; __global const double* row = in + (idx/len)*len;
  double acc = 0.0; for (long k = 0; k < klen; ++k) { long bi = npp_bidx(i+k-anchor, len, mode); acc += w[k]*((bi < 0) ? cval : row[bi]); }
  out[idx] = acc;
}
)CLC";

constexpr int TILE = 16;

struct CLState {
  bool ok = false;
  bool fp64 = false;
  cl_context ctx = nullptr;
  cl_command_queue queue = nullptr;
  cl_kernel binop_f = nullptr, binop_d = nullptr, unop_f = nullptr, unop_d = nullptr,
            reduce_f = nullptr, reduce_d = nullptr, gemm_f = nullptr, gemm_d = nullptr,
            spmv_f = nullptr, spmv_d = nullptr, pdist_f = nullptr, pdist_d = nullptr,
            corr_f = nullptr, corr_d = nullptr;

  CLState() { init(); }

  void init() {
    cl_platform_id plat = nullptr; cl_uint np = 0;
    if (clGetPlatformIDs(1, &plat, &np) != NUMPP_CL_SUCCESS || np == 0) return;
    cl_device_id dev = nullptr; cl_uint nd = 0;
    if (clGetDeviceIDs(plat, NUMPP_CL_DEVICE_TYPE_GPU, 1, &dev, &nd) != NUMPP_CL_SUCCESS || nd == 0) return;
    char ext[8192] = {0};
    clGetDeviceInfo(dev, NUMPP_CL_DEVICE_EXTENSIONS, sizeof(ext), ext, nullptr);
    fp64 = std::string(ext).find("cl_khr_fp64") != std::string::npos;
    cl_int err = 0;
    ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (!ctx) return;
    queue = clCreateCommandQueue(ctx, dev, 0, &err);
    if (!queue) return;
    const char* src = kProgramSrc; size_t len = std::char_traits<char>::length(src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (!prog) return;
    if (clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr) != NUMPP_CL_SUCCESS) return;
    binop_f = clCreateKernel(prog, "binop_f", &err);
    unop_f = clCreateKernel(prog, "unop_f", &err);
    reduce_f = clCreateKernel(prog, "reduce_f", &err);
    gemm_f = clCreateKernel(prog, "gemm_f", &err);
    spmv_f = clCreateKernel(prog, "spmv_f", &err);
    pdist_f = clCreateKernel(prog, "pdist_f", &err);
    corr_f = clCreateKernel(prog, "corr1d_f", &err);
    if (fp64) {
      binop_d = clCreateKernel(prog, "binop_d", &err);
      unop_d = clCreateKernel(prog, "unop_d", &err);
      reduce_d = clCreateKernel(prog, "reduce_d", &err);
      gemm_d = clCreateKernel(prog, "gemm_d", &err);
      spmv_d = clCreateKernel(prog, "spmv_d", &err);
      pdist_d = clCreateKernel(prog, "pdist_d", &err);
      corr_d = clCreateKernel(prog, "corr1d_d", &err);
    }
    ok = binop_f && unop_f && reduce_f && gemm_f && spmv_f && pdist_f && corr_f;
  }
};

CLState& state() { static CLState s; return s; }

// ---- device-buffer reuse pool (avoids clCreateBuffer/clReleaseMemObject per call) ----
struct Buf { cl_mem mem = nullptr; size_t cap = 0; };
struct DevPool {
  std::mutex mu;
  std::vector<Buf> free_;
  Buf acquire(size_t bytes) {
    std::lock_guard<std::mutex> lk(mu);
    for (size_t i = 0; i < free_.size(); ++i) {
      if (free_[i].cap >= bytes) { Buf b = free_[i]; free_.erase(free_.begin() + i); return b; }
    }
    cl_int err = 0;
    Buf b; b.mem = clCreateBuffer(state().ctx, NUMPP_CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (b.mem) b.cap = bytes;
    return b;
  }
  void release(Buf b) {
    if (!b.mem) return;
    std::lock_guard<std::mutex> lk(mu);
    if (free_.size() < 32) free_.push_back(b);
    else clReleaseMemObject(b.mem);
  }
};
DevPool& pool() { static DevPool p; return p; }

bool run_elementwise(cl_kernel k, int op, int64_t n, size_t elem, const void* a, const void* b, void* out) {
  CLState& s = state();
  const size_t bytes = static_cast<size_t>(n) * elem;
  Buf ba = pool().acquire(bytes), bo = pool().acquire(bytes);
  Buf bb = b ? pool().acquire(bytes) : Buf{};
  cl_int e = NUMPP_CL_SUCCESS;
  if (!ba.mem || !bo.mem || (b && !bb.mem)) { e = -1; }
  else {
    clEnqueueWriteBuffer(s.queue, ba.mem, NUMPP_CL_TRUE, 0, bytes, a, 0, nullptr, nullptr);
    if (b) clEnqueueWriteBuffer(s.queue, bb.mem, NUMPP_CL_TRUE, 0, bytes, b, 0, nullptr, nullptr);
    cl_uint i = 0;
    clSetKernelArg(k, i++, sizeof(int), &op);
    clSetKernelArg(k, i++, sizeof(cl_mem), &ba.mem);
    if (b) clSetKernelArg(k, i++, sizeof(cl_mem), &bb.mem);
    clSetKernelArg(k, i++, sizeof(cl_mem), &bo.mem);
    size_t gws = static_cast<size_t>(n);
    e = clEnqueueNDRangeKernel(s.queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (e == NUMPP_CL_SUCCESS) e = clEnqueueReadBuffer(s.queue, bo.mem, NUMPP_CL_TRUE, 0, bytes, out, 0, nullptr, nullptr);
  }
  pool().release(ba); pool().release(bb); pool().release(bo);
  return e == NUMPP_CL_SUCCESS;
}

template <class T>
bool run_reduce(cl_kernel k, int op, int64_t n, const void* a, void* out) {
  CLState& s = state();
  const size_t P = 4096;
  Buf ba = pool().acquire((size_t)n * sizeof(T)), bp = pool().acquire(P * sizeof(T));
  std::vector<T> partial(P);
  cl_int e = NUMPP_CL_SUCCESS;
  if (!ba.mem || !bp.mem) { e = -1; }
  else {
    clEnqueueWriteBuffer(s.queue, ba.mem, NUMPP_CL_TRUE, 0, (size_t)n * sizeof(T), a, 0, nullptr, nullptr);
    long nn = static_cast<long>(n);
    cl_uint i = 0;
    clSetKernelArg(k, i++, sizeof(int), &op);
    clSetKernelArg(k, i++, sizeof(cl_mem), &ba.mem);
    clSetKernelArg(k, i++, sizeof(long), &nn);
    clSetKernelArg(k, i++, sizeof(cl_mem), &bp.mem);
    size_t gws = P;
    e = clEnqueueNDRangeKernel(s.queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (e == NUMPP_CL_SUCCESS) e = clEnqueueReadBuffer(s.queue, bp.mem, NUMPP_CL_TRUE, 0, P * sizeof(T), partial.data(), 0, nullptr, nullptr);
  }
  pool().release(ba); pool().release(bp);
  if (e != NUMPP_CL_SUCCESS) return false;
  T acc = (op == kGProd) ? T(1) : T(0);
  for (size_t j = 0; j < P; ++j) acc = (op == kGProd) ? acc * partial[j] : acc + partial[j];
  *static_cast<T*>(out) = acc;
  return true;
}

template <class T>
bool run_gemm(cl_kernel k, int64_t m, int64_t n, int64_t kk, const void* a, const void* b, void* out) {
  CLState& s = state();
  Buf ba = pool().acquire((size_t)m * kk * sizeof(T));
  Buf bb = pool().acquire((size_t)kk * n * sizeof(T));
  Buf bo = pool().acquire((size_t)m * n * sizeof(T));
  cl_int e = NUMPP_CL_SUCCESS;
  if (!ba.mem || !bb.mem || !bo.mem) { e = -1; }
  else {
    clEnqueueWriteBuffer(s.queue, ba.mem, NUMPP_CL_TRUE, 0, (size_t)m * kk * sizeof(T), a, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(s.queue, bb.mem, NUMPP_CL_TRUE, 0, (size_t)kk * n * sizeof(T), b, 0, nullptr, nullptr);
    long M = (long)m, N = (long)n, K = (long)kk;
    cl_uint i = 0;
    clSetKernelArg(k, i++, sizeof(long), &M); clSetKernelArg(k, i++, sizeof(long), &N); clSetKernelArg(k, i++, sizeof(long), &K);
    clSetKernelArg(k, i++, sizeof(cl_mem), &ba.mem); clSetKernelArg(k, i++, sizeof(cl_mem), &bb.mem); clSetKernelArg(k, i++, sizeof(cl_mem), &bo.mem);
    size_t lws[2] = {TILE, TILE};
    size_t gws[2] = {((size_t)(n + TILE - 1) / TILE) * TILE, ((size_t)(m + TILE - 1) / TILE) * TILE};
    e = clEnqueueNDRangeKernel(s.queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    if (e == NUMPP_CL_SUCCESS) e = clEnqueueReadBuffer(s.queue, bo.mem, NUMPP_CL_TRUE, 0, (size_t)m * n * sizeof(T), out, 0, nullptr, nullptr);
  }
  pool().release(ba); pool().release(bb); pool().release(bo);
  return e == NUMPP_CL_SUCCESS;
}

template <class T>
bool run_spmv(cl_kernel k, int64_t rows, int64_t cols, int64_t nnz, const int64_t* indptr,
              const int64_t* indices, const void* data, const void* x, void* y) {
  CLState& s = state();
  Buf bip = pool().acquire((size_t)(rows + 1) * sizeof(int64_t)), bix = pool().acquire((size_t)nnz * sizeof(int64_t));
  Buf bd = pool().acquire((size_t)nnz * sizeof(T)), bx = pool().acquire((size_t)cols * sizeof(T)), by = pool().acquire((size_t)rows * sizeof(T));
  cl_int e = NUMPP_CL_SUCCESS;
  if (!bip.mem || !bix.mem || !bd.mem || !bx.mem || !by.mem) { e = -1; }
  else {
    clEnqueueWriteBuffer(s.queue, bip.mem, NUMPP_CL_TRUE, 0, (size_t)(rows + 1) * sizeof(int64_t), indptr, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(s.queue, bix.mem, NUMPP_CL_TRUE, 0, (size_t)nnz * sizeof(int64_t), indices, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(s.queue, bd.mem, NUMPP_CL_TRUE, 0, (size_t)nnz * sizeof(T), data, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(s.queue, bx.mem, NUMPP_CL_TRUE, 0, (size_t)cols * sizeof(T), x, 0, nullptr, nullptr);
    long R = (long)rows; cl_uint i = 0;
    clSetKernelArg(k, i++, sizeof(long), &R);
    clSetKernelArg(k, i++, sizeof(cl_mem), &bip.mem); clSetKernelArg(k, i++, sizeof(cl_mem), &bix.mem);
    clSetKernelArg(k, i++, sizeof(cl_mem), &bd.mem); clSetKernelArg(k, i++, sizeof(cl_mem), &bx.mem); clSetKernelArg(k, i++, sizeof(cl_mem), &by.mem);
    size_t gws = (size_t)rows;
    e = clEnqueueNDRangeKernel(s.queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (e == NUMPP_CL_SUCCESS) e = clEnqueueReadBuffer(s.queue, by.mem, NUMPP_CL_TRUE, 0, (size_t)rows * sizeof(T), y, 0, nullptr, nullptr);
  }
  pool().release(bip); pool().release(bix); pool().release(bd); pool().release(bx); pool().release(by);
  return e == NUMPP_CL_SUCCESS;
}
template <class T>
bool run_pdist(cl_kernel k, int64_t m, int64_t n, int64_t dim, const void* A, const void* B, void* D) {
  CLState& s = state();
  Buf ba = pool().acquire((size_t)m * dim * sizeof(T)), bb = pool().acquire((size_t)n * dim * sizeof(T)), bd = pool().acquire((size_t)m * n * sizeof(T));
  cl_int e = NUMPP_CL_SUCCESS;
  if (!ba.mem || !bb.mem || !bd.mem) { e = -1; }
  else {
    clEnqueueWriteBuffer(s.queue, ba.mem, NUMPP_CL_TRUE, 0, (size_t)m * dim * sizeof(T), A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(s.queue, bb.mem, NUMPP_CL_TRUE, 0, (size_t)n * dim * sizeof(T), B, 0, nullptr, nullptr);
    long M = (long)m, N = (long)n, Dm = (long)dim; cl_uint i = 0;
    clSetKernelArg(k, i++, sizeof(long), &M); clSetKernelArg(k, i++, sizeof(long), &N); clSetKernelArg(k, i++, sizeof(long), &Dm);
    clSetKernelArg(k, i++, sizeof(cl_mem), &ba.mem); clSetKernelArg(k, i++, sizeof(cl_mem), &bb.mem); clSetKernelArg(k, i++, sizeof(cl_mem), &bd.mem);
    size_t gws = (size_t)(m * n);
    e = clEnqueueNDRangeKernel(s.queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (e == NUMPP_CL_SUCCESS) e = clEnqueueReadBuffer(s.queue, bd.mem, NUMPP_CL_TRUE, 0, (size_t)m * n * sizeof(T), D, 0, nullptr, nullptr);
  }
  pool().release(ba); pool().release(bb); pool().release(bd);
  return e == NUMPP_CL_SUCCESS;
}
template <class T>
bool run_corr1d(cl_kernel k, int64_t lines, int64_t len, const void* in, const void* w, int64_t klen,
                int64_t origin, int mode, double cval, void* out) {
  CLState& s = state();
  const size_t nbytes = (size_t)lines * len * sizeof(T);
  Buf bin = pool().acquire(nbytes), bw = pool().acquire((size_t)klen * sizeof(T)), bout = pool().acquire(nbytes);
  cl_int e = NUMPP_CL_SUCCESS;
  if (!bin.mem || !bw.mem || !bout.mem) { e = -1; }
  else {
    clEnqueueWriteBuffer(s.queue, bin.mem, NUMPP_CL_TRUE, 0, nbytes, in, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(s.queue, bw.mem, NUMPP_CL_TRUE, 0, (size_t)klen * sizeof(T), w, 0, nullptr, nullptr);
    long L = (long)lines, Ln = (long)len, K = (long)klen, anchor = (long)(klen / 2 + origin);
    T cv = (T)cval; cl_uint i = 0;
    clSetKernelArg(k, i++, sizeof(long), &L); clSetKernelArg(k, i++, sizeof(long), &Ln);
    clSetKernelArg(k, i++, sizeof(cl_mem), &bin.mem); clSetKernelArg(k, i++, sizeof(cl_mem), &bw.mem);
    clSetKernelArg(k, i++, sizeof(long), &K); clSetKernelArg(k, i++, sizeof(long), &anchor);
    clSetKernelArg(k, i++, sizeof(int), &mode); clSetKernelArg(k, i++, sizeof(T), &cv);
    clSetKernelArg(k, i++, sizeof(cl_mem), &bout.mem);
    size_t gws = (size_t)(lines * len);
    e = clEnqueueNDRangeKernel(s.queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (e == NUMPP_CL_SUCCESS) e = clEnqueueReadBuffer(s.queue, bout.mem, NUMPP_CL_TRUE, 0, nbytes, out, 0, nullptr, nullptr);
  }
  pool().release(bin); pool().release(bw); pool().release(bout);
  return e == NUMPP_CL_SUCCESS;
}

bool ew_binary(int op, DTypeId dt, int64_t n, const void* a, const void* b, void* out) {
  if (op < kGAdd || op > kGDiv || !state().ok) return false;
  if (dt == DTypeId::Float32) return run_elementwise(state().binop_f, op, n, sizeof(float), a, b, out);
  if (dt == DTypeId::Float64 && state().fp64) return run_elementwise(state().binop_d, op, n, sizeof(double), a, b, out);
  return false;
}
bool ew_unary(int op, DTypeId dt, int64_t n, const void* a, void* out) {
  if ((op != kGNeg && op != kGSqrt) || !state().ok) return false;
  if (dt == DTypeId::Float32) return run_elementwise(state().unop_f, op, n, sizeof(float), a, nullptr, out);
  if (dt == DTypeId::Float64 && state().fp64) return run_elementwise(state().unop_d, op, n, sizeof(double), a, nullptr, out);
  return false;
}
bool reduce_impl(int op, DTypeId dt, int64_t n, const void* a, void* out) {
  if ((op != kGSum && op != kGProd) || !state().ok) return false;
  if (dt == DTypeId::Float32) return run_reduce<float>(state().reduce_f, op, n, a, out);
  if (dt == DTypeId::Float64 && state().fp64) return run_reduce<double>(state().reduce_d, op, n, a, out);
  return false;
}
bool gemm_impl(DTypeId dt, int64_t m, int64_t n, int64_t k, const void* A, const void* B, void* C) {
  if (!state().ok) return false;
  if (dt == DTypeId::Float32) return run_gemm<float>(state().gemm_f, m, n, k, A, B, C);
  if (dt == DTypeId::Float64 && state().fp64) return run_gemm<double>(state().gemm_d, m, n, k, A, B, C);
  return false;
}
bool csr_spmv_impl(DTypeId dt, int64_t rows, int64_t cols, int64_t nnz, const int64_t* indptr,
                   const int64_t* indices, const void* data, const void* x, void* y) {
  if (!state().ok) return false;
  if (dt == DTypeId::Float32) return run_spmv<float>(state().spmv_f, rows, cols, nnz, indptr, indices, data, x, y);
  if (dt == DTypeId::Float64 && state().fp64) return run_spmv<double>(state().spmv_d, rows, cols, nnz, indptr, indices, data, x, y);
  return false;
}
bool pairwise_sqdist_impl(DTypeId dt, int64_t m, int64_t n, int64_t dim, const void* A, const void* B, void* D) {
  if (!state().ok) return false;
  if (dt == DTypeId::Float32) return run_pdist<float>(state().pdist_f, m, n, dim, A, B, D);
  if (dt == DTypeId::Float64 && state().fp64) return run_pdist<double>(state().pdist_d, m, n, dim, A, B, D);
  return false;
}
bool separable_corr1d_impl(DTypeId dt, int64_t lines, int64_t len, const void* in, const void* weights,
                           int64_t klen, int64_t origin, int mode, double cval, void* out) {
  if (!state().ok) return false;
  if (dt == DTypeId::Float32) return run_corr1d<float>(state().corr_f, lines, len, in, weights, klen, origin, mode, cval, out);
  if (dt == DTypeId::Float64 && state().fp64) return run_corr1d<double>(state().corr_d, lines, len, in, weights, klen, origin, mode, cval, out);
  return false;
}

const GpuVTable g_vtable{"opencl", &ew_binary, &ew_unary, &reduce_impl, &gemm_impl,
                         &csr_spmv_impl, &pairwise_sqdist_impl, &separable_corr1d_impl};

}  // namespace

const GpuVTable* gpu_vtable() { return state().ok ? &g_vtable : nullptr; }

}  // namespace numpp
