#include "numpp/backend/gpu_vtable.hpp"

#include "opencl_min.h"

#include <mutex>
#include <string>
#include <vector>

// Real OpenCL GPU backend, compiled only when NUMPP_WITH_OPENCL=ON. It implements
// the GPU vtable with kernels that run on an OpenCL GPU device (e.g. NVIDIA via
// the ICD loader). float32/float64 only; everything else returns false so the
// ufunc dispatcher falls back to the portable CPU kernel.
//
// Arithmetic (add/sub/mul/div), negate and sqrt are IEEE-754 operations that are
// bit-identical to the CPU path; exp (transcendental, ULP-approximate on device)
// and complex dtypes are intentionally declined to the CPU kernel.

namespace numpp {
namespace {

// OpenCL program: generic op-coded kernels for float (f) and double (d).
const char* kProgramSrc = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
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
  int i = get_global_id(0); float x = a[i];
  o[i] = (op == 1) ? sqrt(x) : -x;
}
__kernel void unop_d(int op, __global const double* a, __global double* o) {
  int i = get_global_id(0); double x = a[i];
  o[i] = (op == 1) ? sqrt(x) : -x;
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
)CLC";

struct CLState {
  bool ok = false;
  bool fp64 = false;
  cl_context ctx = nullptr;
  cl_command_queue queue = nullptr;
  cl_kernel binop_f = nullptr, binop_d = nullptr, unop_f = nullptr, unop_d = nullptr,
            reduce_f = nullptr, reduce_d = nullptr;

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
    if (fp64) {
      binop_d = clCreateKernel(prog, "binop_d", &err);
      unop_d = clCreateKernel(prog, "unop_d", &err);
      reduce_d = clCreateKernel(prog, "reduce_d", &err);
    }
    ok = binop_f && unop_f && reduce_f;
  }
};

CLState& state() {
  static CLState s;
  return s;
}

cl_mem buf(cl_mem_flags flags, size_t bytes, void* host) {
  cl_int err = 0;
  return clCreateBuffer(state().ctx, flags, bytes, host, &err);
}

// Run an op-coded elementwise kernel over n elements of `elem` bytes each.
bool run_elementwise(cl_kernel k, int op, int64_t n, size_t elem, const void* a, const void* b, void* out) {
  CLState& s = state();
  const size_t bytes = static_cast<size_t>(n) * elem;
  cl_mem ba = buf(NUMPP_CL_MEM_READ_ONLY | NUMPP_CL_MEM_COPY_HOST_PTR, bytes, const_cast<void*>(a));
  cl_mem bb = b ? buf(NUMPP_CL_MEM_READ_ONLY | NUMPP_CL_MEM_COPY_HOST_PTR, bytes, const_cast<void*>(b)) : nullptr;
  cl_mem bo = buf(NUMPP_CL_MEM_WRITE_ONLY, bytes, nullptr);
  if (!ba || !bo || (b && !bb)) { return false; }
  cl_uint i = 0;
  clSetKernelArg(k, i++, sizeof(int), &op);
  clSetKernelArg(k, i++, sizeof(cl_mem), &ba);
  if (b) clSetKernelArg(k, i++, sizeof(cl_mem), &bb);
  clSetKernelArg(k, i++, sizeof(cl_mem), &bo);
  size_t gws = static_cast<size_t>(n);
  cl_int e = clEnqueueNDRangeKernel(s.queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
  if (e == NUMPP_CL_SUCCESS) e = clEnqueueReadBuffer(s.queue, bo, NUMPP_CL_TRUE, 0, bytes, out, 0, nullptr, nullptr);
  clReleaseMemObject(ba); if (bb) clReleaseMemObject(bb); clReleaseMemObject(bo);
  return e == NUMPP_CL_SUCCESS;
}

template <class T>
bool run_reduce(cl_kernel k, int op, int64_t n, const void* a, void* out) {
  CLState& s = state();
  const size_t P = 4096;  // number of partial accumulators
  cl_mem ba = buf(NUMPP_CL_MEM_READ_ONLY | NUMPP_CL_MEM_COPY_HOST_PTR, static_cast<size_t>(n) * sizeof(T), const_cast<void*>(a));
  cl_mem bp = buf(NUMPP_CL_MEM_WRITE_ONLY, P * sizeof(T), nullptr);
  if (!ba || !bp) return false;
  long nn = static_cast<long>(n);
  cl_uint i = 0;
  clSetKernelArg(k, i++, sizeof(int), &op);
  clSetKernelArg(k, i++, sizeof(cl_mem), &ba);
  clSetKernelArg(k, i++, sizeof(long), &nn);
  clSetKernelArg(k, i++, sizeof(cl_mem), &bp);
  size_t gws = P;
  std::vector<T> partial(P);
  cl_int e = clEnqueueNDRangeKernel(s.queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
  if (e == NUMPP_CL_SUCCESS) e = clEnqueueReadBuffer(s.queue, bp, NUMPP_CL_TRUE, 0, P * sizeof(T), partial.data(), 0, nullptr, nullptr);
  clReleaseMemObject(ba); clReleaseMemObject(bp);
  if (e != NUMPP_CL_SUCCESS) return false;
  T acc = (op == kGProd) ? T(1) : T(0);
  for (size_t j = 0; j < P; ++j) acc = (op == kGProd) ? acc * partial[j] : acc + partial[j];
  *static_cast<T*>(out) = acc;
  return true;
}

bool ew_binary(int op, DTypeId dt, int64_t n, const void* a, const void* b, void* out) {
  if (op < kGAdd || op > kGDiv || !state().ok) return false;
  if (dt == DTypeId::Float32) return run_elementwise(state().binop_f, op, n, sizeof(float), a, b, out);
  if (dt == DTypeId::Float64 && state().fp64) return run_elementwise(state().binop_d, op, n, sizeof(double), a, b, out);
  return false;
}
bool ew_unary(int op, DTypeId dt, int64_t n, const void* a, void* out) {
  if ((op != kGNeg && op != kGSqrt) || !state().ok) return false;  // exp declined: ULP-approximate on device
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

const GpuVTable g_vtable{"opencl", &ew_binary, &ew_unary, &reduce_impl};

}  // namespace

const GpuVTable* gpu_vtable() { return state().ok ? &g_vtable : nullptr; }

}  // namespace numpp
