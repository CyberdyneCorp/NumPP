#include "numpp/backend/backend.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "numpp/backend/blas_vtable.hpp"
#include "numpp/backend/gpu_vtable.hpp"
#include "numpp/backend/config.hpp"
#include "numpp/core/shape.hpp"          // normalize_axis (correlate1d)

#include "scypp_kernels.hpp"             // shared CPU reference kernels

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
  // Compiled in AND a device actually registered a vtable (gpu_vtable() probes
  // for a real device; null means compiled but no device present).
  return backend_compiled(b) && gpu_vtable() != nullptr;
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

  // Explicitly-targeted GPU GEMM (forced backend or NUMPP_GPU_TARGET). Auto stays
  // on BLAS/CPU. The device gemm accumulates in p-order so it is bitwise-equal to
  // the CPU gemm for float32/float64.
  if (is_gpu(target) && gpu_vtable() && gpu_vtable()->gemm && (dt == kFloat32 || dt == kFloat64)) {
    ndarray ac = a.astype(dt).ascontiguousarray();
    ndarray bc = b.astype(dt).ascontiguousarray();
    ndarray c(Shape{m, n}, dt, Order::C);
    if (gpu_vtable()->gemm(dt.id(), m, n, k, ac.bytes(), bc.bytes(), c.bytes())) {
      t_last = target;
      return c;
    }
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

// ---- ScyPP acceleration primitives -----------------------------------------
namespace {

int64_t device_min() {
  if (const char* e = std::getenv("NUMPP_GPU_MIN")) {
    char* end = nullptr;
    long long v = std::strtoll(e, &end, 10);
    if (end != e && v >= 0) return v;
  }
  return 1 << 16;
}
bool is_gpu_backend(Backend b) {
  return b == Backend::Metal || b == Backend::Vulkan || b == Backend::Cuda ||
         b == Backend::OpenCL || b == Backend::Device;
}
// Decide whether to attempt the device slot. `slot_present` is (vtable && slot).
// A forced GPU backend with no usable slot is an error; Auto/Device use the slot
// when present and the problem clears the size threshold.
bool try_device(Backend forced, bool slot_present, int64_t work) {
  if (forced == Backend::Cpu) return false;
  if (!slot_present) {
    if (is_gpu_backend(forced))
      throw not_implemented_error(std::string("backend '") + backend_name(forced) +
                                  "' is not available for this op");
    return false;
  }
  if (is_gpu_backend(forced)) return true;  // explicit device: bypass threshold
  return work >= device_min();              // Auto
}
// float32 only when both operands are float32; otherwise float64.
DType pick_float(DType a, DType b) {
  return (a == kFloat32 && b == kFloat32) ? kFloat32 : kFloat64;
}

}  // namespace

ndarray csr_spmv(const ndarray& indptr, const ndarray& indices, const ndarray& data,
                 const ndarray& x, Backend forced) {
  if (indptr.ndim() != 1 || indices.ndim() != 1 || data.ndim() != 1 || x.ndim() != 1)
    throw value_error("csr_spmv: indptr/indices/data/x must be 1-D");
  if (indptr.size() < 1) throw value_error("csr_spmv: indptr must have length rows+1");
  const int64_t rows = indptr.size() - 1;
  const int64_t nnz = data.size();
  const DType dt = pick_float(data.dtype(), x.dtype());
  ndarray ip = indptr.astype(kInt64).ascontiguousarray();
  ndarray ix = indices.astype(kInt64).ascontiguousarray();
  ndarray dd = data.astype(dt).ascontiguousarray();
  ndarray xx = x.astype(dt).ascontiguousarray();
  ndarray y(Shape{rows}, dt, Order::C);

  const GpuVTable* vt = gpu_vtable();
  const bool slot = vt && vt->csr_spmv;
  if (try_device(forced, slot, nnz) &&
      vt->csr_spmv(dt.id(), rows, x.size(), nnz, ip.typed_data<int64_t>(), ix.typed_data<int64_t>(),
                   dd.bytes(), xx.bytes(), y.bytes())) {
    t_last = Backend::Device;
    return y;
  }
  t_last = Backend::Cpu;
  if (dt == kFloat32)
    scypp_cpu::csr_spmv<float>(rows, ip.typed_data<int64_t>(), ix.typed_data<int64_t>(),
                               dd.typed_data<float>(), xx.typed_data<float>(), y.typed_data<float>());
  else
    scypp_cpu::csr_spmv<double>(rows, ip.typed_data<int64_t>(), ix.typed_data<int64_t>(),
                                dd.typed_data<double>(), xx.typed_data<double>(), y.typed_data<double>());
  return y;
}

ndarray cdist_euclidean(const ndarray& A, const ndarray& B, bool squared, Backend forced) {
  if (A.ndim() != 2 || B.ndim() != 2) throw value_error("cdist_euclidean: A and B must be 2-D");
  if (A.shape()[1] != B.shape()[1]) throw value_error("cdist_euclidean: dim mismatch");
  const int64_t m = A.shape()[0], n = B.shape()[0], dim = A.shape()[1];
  const DType dt = pick_float(A.dtype(), B.dtype());
  ndarray Ac = A.astype(dt).ascontiguousarray();
  ndarray Bc = B.astype(dt).ascontiguousarray();
  ndarray D(Shape{m, n}, dt, Order::C);

  const GpuVTable* vt = gpu_vtable();
  const bool slot = vt && vt->pairwise_sqdist;
  bool on_device = false;
  if (try_device(forced, slot, m * n * dim) &&
      vt->pairwise_sqdist(dt.id(), m, n, dim, Ac.bytes(), Bc.bytes(), D.bytes())) {
    t_last = Backend::Device;
    on_device = true;
  }
  if (!on_device) {
    t_last = Backend::Cpu;
    if (dt == kFloat32)
      scypp_cpu::pairwise_sqdist<float>(m, n, dim, Ac.typed_data<float>(), Bc.typed_data<float>(), D.typed_data<float>());
    else
      scypp_cpu::pairwise_sqdist<double>(m, n, dim, Ac.typed_data<double>(), Bc.typed_data<double>(), D.typed_data<double>());
  }
  if (squared) return D;
  // euclidean = sqrt(squared), applied on the host for both paths.
  if (dt == kFloat32) { float* p = D.typed_data<float>(); for (int64_t i = 0; i < m * n; ++i) p[i] = std::sqrt(p[i]); }
  else { double* p = D.typed_data<double>(); for (int64_t i = 0; i < m * n; ++i) p[i] = std::sqrt(p[i]); }
  return D;
}

ndarray correlate1d(const ndarray& input, const ndarray& weights, int64_t axis,
                    int64_t origin, FilterMode mode, double cval, Backend forced) {
  if (weights.ndim() != 1 || weights.size() == 0) throw value_error("correlate1d: weights must be 1-D, non-empty");
  const int64_t nd = input.ndim();
  if (nd == 0) throw value_error("correlate1d: input must be at least 1-D");
  const int64_t ax = normalize_axis(axis, nd);
  const DType dt = (input.dtype() == kFloat32 && weights.dtype() == kFloat32) ? kFloat32 : kFloat64;
  const int64_t len = input.shape()[ax];
  const int64_t klen = weights.size();

  // Move the filter axis last and flatten to (lines, len) contiguous rows.
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < nd; ++i) if (i != ax) perm.push_back(i);
  perm.push_back(ax);
  ndarray moved = input.transpose(perm).astype(dt).ascontiguousarray();
  ndarray w = weights.astype(dt).ascontiguousarray();
  const int64_t lines = len ? moved.size() / len : 0;
  ndarray out_moved(moved.shape(), dt, Order::C);
  const int imode = static_cast<int>(mode);

  const GpuVTable* vt = gpu_vtable();
  const bool slot = vt && vt->separable_corr1d;
  bool on_device = false;
  if (try_device(forced, slot, moved.size()) &&
      vt->separable_corr1d(dt.id(), lines, len, moved.bytes(), w.bytes(), klen, origin, imode, cval, out_moved.bytes())) {
    t_last = Backend::Device;
    on_device = true;
  }
  if (!on_device) {
    t_last = Backend::Cpu;
    if (dt == kFloat32)
      scypp_cpu::separable_corr1d<float>(lines, len, moved.typed_data<float>(), w.typed_data<float>(),
                                         klen, origin, imode, cval, out_moved.typed_data<float>());
    else
      scypp_cpu::separable_corr1d<double>(lines, len, moved.typed_data<double>(), w.typed_data<double>(),
                                          klen, origin, imode, cval, out_moved.typed_data<double>());
  }
  // Invert the permutation back to the original axis order.
  std::vector<int64_t> inv(nd);
  for (int64_t i = 0; i < nd; ++i) inv[perm[i]] = i;
  return out_moved.transpose(inv).ascontiguousarray();
}

}  // namespace numpp
