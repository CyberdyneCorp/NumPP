#include "numpp/backend/backend.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

#include "numpp/backend/blas_vtable.hpp"
#include "numpp/backend/config.hpp"

namespace numpp {
namespace {

thread_local Backend t_last = Backend::Cpu;

int64_t gemm_threshold() {
  if (const char* e = std::getenv("NUMPP_GEMM_MIN")) {
    char* end = nullptr;
    long long v = std::strtoll(e, &end, 10);
    if (end != e && v >= 0) return v;
  }
  return 1 << 18;  // 262144 flops-ish, matching matlab_llvm's default
}

// Parse NUMPP_GPU_TARGET. Returns the requested backend; Auto if unset/"auto".
Backend env_target() {
  const char* e = std::getenv("NUMPP_GPU_TARGET");
  if (!e) return Backend::Auto;
  std::string s(e);
  if (s == "cpu") return Backend::Cpu;
  if (s == "metal") return Backend::Metal;
  if (s == "vulkan") return Backend::Vulkan;
  if (s == "cuda") return Backend::Cuda;
  if (s == "opencl") return Backend::OpenCL;
  return Backend::Auto;  // "auto" or unknown
}

bool backend_compiled(Backend b) {
  switch (b) {
    case Backend::Metal:  return NUMPP_WITH_METAL;
    case Backend::Vulkan: return NUMPP_WITH_VULKAN;
    case Backend::Cuda:   return NUMPP_WITH_CUDA;
    case Backend::OpenCL: return NUMPP_WITH_OPENCL;
    default: return false;
  }
}

// Generic naive CPU GEMM: C[m,n] = A[m,k]·B[k,n]. Computed in an arithmetic
// `compute` dtype (half -> float32, bool -> int64) then cast to the result dtype.
ndarray cpu_matmul(const ndarray& a, const ndarray& b, DType dt) {
  DType compute = dt;
  if (dt == kFloat16) compute = kFloat32;
  else if (dt == kBool) compute = kInt64;
  const int64_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
  ndarray ac = a.astype(compute).ascontiguousarray();
  ndarray bc = b.astype(compute).ascontiguousarray();
  ndarray c = ndarray(Shape{m, n}, compute, Order::C);
  visit_dtype(compute.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bool>) {
      // Unreachable: compute dtype is never half/bool. Present only so the
      // visit_dtype switch compiles for every branch.
    } else {
      const T* A = ac.typed_data<T>();
      const T* B = bc.typed_data<T>();
      T* C = c.typed_data<T>();
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) C[i * n + j] = T{};
        for (int64_t p = 0; p < k; ++p) {
          const T aip = A[i * k + p];
          const T* brow = B + p * n;
          T* crow = C + i * n;
          for (int64_t j = 0; j < n; ++j) crow[j] += aip * brow[j];
        }
      }
    }
  });
  return compute == dt ? c : c.astype(dt);
}

}  // namespace

const char* backend_name(Backend b) {
  switch (b) {
    case Backend::Auto: return "auto";
    case Backend::Cpu: return "cpu";
    case Backend::Blas: return "blas";
    case Backend::Device: return "device";
    case Backend::Metal: return "metal";
    case Backend::Vulkan: return "vulkan";
    case Backend::Cuda: return "cuda";
    case Backend::OpenCL: return "opencl";
  }
  return "?";
}

CapabilityRegistry::CapabilityRegistry() {
  has_blas_ = (blas_vtable() != nullptr);
  has_lapack_ = NUMPP_WITH_LAPACK != 0;
}

const CapabilityRegistry& CapabilityRegistry::instance() {
  static const CapabilityRegistry reg;  // thread-safe static init (C++11+)
  return reg;
}

bool CapabilityRegistry::gpu_available(Backend b) const {
  // GPU backends are present as dispatch slots; device probing lands with the
  // GPU kernels (roadmap Phase 10). Compiled-in but no device -> unavailable.
  return backend_compiled(b) && false;
}

Backend last_backend() { return t_last; }
void set_last_backend(Backend b) { t_last = b; }

ndarray matmul(const ndarray& a, const ndarray& b, Backend forced) {
  if (a.ndim() != 2 || b.ndim() != 2) throw value_error("matmul requires 2-D arrays");
  if (a.shape()[1] != b.shape()[0]) throw value_error("matmul: shape mismatch");
  const DType dt = result_type(a.dtype(), b.dtype());
  const int64_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
  const int64_t work = m * n * k;

  const auto& cap = CapabilityRegistry::instance();
  Backend target = forced != Backend::Auto ? forced : env_target();

  // Explicitly requesting a GPU backend that was not compiled in is an error;
  // `auto` (unset target) silently degrades to CPU instead.
  auto is_gpu = [](Backend b) {
    return b == Backend::Metal || b == Backend::Vulkan || b == Backend::Cuda || b == Backend::OpenCL;
  };
  if (is_gpu(target) && !cap.gpu_available(target)) {
    throw not_implemented_error(std::string("backend '") + backend_name(target) +
                                "' is not available in this build");
  }

  const bool blas_dtype = (dt == kFloat32 || dt == kFloat64 || dt == kComplex64 || dt == kComplex128);

  bool use_blas = false;
  if (forced == Backend::Blas) {
    if (!cap.has_blas()) throw not_implemented_error("BLAS backend not available");
    use_blas = blas_dtype;
  } else if (target == Backend::Cpu) {
    use_blas = false;
  } else if (forced == Backend::Auto && target == Backend::Auto) {
    use_blas = cap.has_blas() && blas_dtype && work >= gemm_threshold();
  }

  if (use_blas && cap.has_blas()) {
    ndarray ac = a.astype(dt).ascontiguousarray();
    ndarray bc = b.astype(dt).ascontiguousarray();
    ndarray c(Shape{m, n}, dt, Order::C);
    if (blas_vtable()->gemm(dt.id(), m, n, k, ac.bytes(), bc.bytes(), c.bytes())) {
      t_last = Backend::Blas;
      return c;
    }
  }
  t_last = Backend::Cpu;
  return cpu_matmul(a, b, dt);
}

}  // namespace numpp
