#pragma once

#include <cstdint>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Which implementation served an operation. `Auto` lets the dispatcher decide.
enum class Backend { Auto, Cpu, Blas, Device, Metal, Vulkan, Cuda, OpenCL };

NUMPP_API const char* backend_name(Backend b);

// Record / query the backend that served the most recent dispatched op (this
// thread). Used by the ufunc layer's GPU dispatch and by tests.
NUMPP_API void set_last_backend(Backend b);

// Runtime capability probe (thread-safe, initialized once). Mirrors the
// weak-linked-backend model: a backend is "available" only if compiled in AND
// (for GPU) a usable device is present.
class NUMPP_API CapabilityRegistry {
 public:
  static const CapabilityRegistry& instance();

  bool has_blas() const { return has_blas_; }
  bool has_lapack() const { return has_lapack_; }
  bool gpu_available(Backend b) const;

 private:
  CapabilityRegistry();
  bool has_blas_ = false;
  bool has_lapack_ = false;
};

// Backend that served the most recent dispatched op on this thread (diagnostics).
NUMPP_API Backend last_backend();

// Matrix product of two 2-D arrays. `forced` overrides dispatch (used by tests);
// `Auto` selects by availability and problem size, always falling back to CPU.
NUMPP_API ndarray matmul(const ndarray& a, const ndarray& b, Backend forced = Backend::Auto);

}  // namespace numpp
