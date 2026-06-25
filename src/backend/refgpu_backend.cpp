#include "numpp/backend/gpu_vtable.hpp"

#include <cmath>
#include <complex>

// CPU-reference "device" backend, compiled only when NUMPP_WITH_REFGPU=ON. It
// implements the GPU vtable by running the same math on the host buffers, so its
// results provably match the portable CPU path. This proves the GPU dispatch
// pipeline end-to-end without real device hardware.

namespace numpp {
namespace {

template <class T>
bool bin(int op, int64_t n, const T* a, const T* b, T* out) {
  for (int64_t i = 0; i < n; ++i) {
    switch (op) {
      case kGAdd: out[i] = a[i] + b[i]; break;
      case kGSub: out[i] = a[i] - b[i]; break;
      case kGMul: out[i] = a[i] * b[i]; break;
      case kGDiv: out[i] = a[i] / b[i]; break;
      default: return false;
    }
  }
  return true;
}
template <class T>
bool un(int op, int64_t n, const T* a, T* out) {
  for (int64_t i = 0; i < n; ++i) {
    switch (op) {
      case kGNeg: out[i] = -a[i]; break;
      case kGSqrt: out[i] = std::sqrt(a[i]); break;
      case kGExp: out[i] = std::exp(a[i]); break;
      default: return false;
    }
  }
  return true;
}
template <class T>
bool red(int op, int64_t n, const T* a, T* out) {
  T acc = op == kGProd ? T(1) : T(0);
  for (int64_t i = 0; i < n; ++i) acc = op == kGProd ? acc * a[i] : acc + a[i];
  *out = acc;
  return true;
}

template <class F>
bool dispatch(DTypeId dt, F&& f) {
  switch (dt) {
    case DTypeId::Float32: return f(float{});
    case DTypeId::Float64: return f(double{});
    case DTypeId::Complex64: return f(std::complex<float>{});
    case DTypeId::Complex128: return f(std::complex<double>{});
    default: return false;
  }
}

bool ew_binary(int op, DTypeId dt, int64_t n, const void* a, const void* b, void* out) {
  return dispatch(dt, [&](auto tag) { using T = decltype(tag); return bin<T>(op, n, static_cast<const T*>(a), static_cast<const T*>(b), static_cast<T*>(out)); });
}
bool ew_unary(int op, DTypeId dt, int64_t n, const void* a, void* out) {
  return dispatch(dt, [&](auto tag) { using T = decltype(tag); return un<T>(op, n, static_cast<const T*>(a), static_cast<T*>(out)); });
}
bool reduce_impl(int op, DTypeId dt, int64_t n, const void* a, void* out) {
  return dispatch(dt, [&](auto tag) { using T = decltype(tag); return red<T>(op, n, static_cast<const T*>(a), static_cast<T*>(out)); });
}

const GpuVTable g_vtable{"refgpu(cpu)", &ew_binary, &ew_unary, &reduce_impl};

}  // namespace

const GpuVTable* gpu_vtable() { return &g_vtable; }

}  // namespace numpp
