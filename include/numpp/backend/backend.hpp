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

// ---- ScyPP acceleration primitives -----------------------------------------
// Low-level device-accelerable kernels whose high-level API lives in ScyPP (the
// C++ SciPy port): sparse SpMV, pairwise distance, separable filtering. They are
// NOT part of NumPP's numpy.* surface — they expose NumPP's shared tiered-
// acceleration substrate so ScyPP can offload through the same GpuVTable. Each
// auto-selects a device backend when one is available and the problem clears the
// size threshold (NUMPP_GPU_MIN), always falling back to the portable CPU kernel
// and recording the choice for last_backend(). `forced` overrides dispatch.

// CSR sparse matrix-vector product y = A·x. A is an (indptr.size()-1) × cols
// sparse matrix in CSR form; x has length cols; returns y of length rows. data/x
// are cast to a common float (float32/float64); indptr/indices are integer.
NUMPP_API ndarray csr_spmv(const ndarray& indptr, const ndarray& indices,
                           const ndarray& data, const ndarray& x,
                           Backend forced = Backend::Auto);

// Pairwise distance matrix between the rows of A (m × dim) and B (n × dim):
// returns D (m × n) of euclidean distances, or squared-euclidean if squared=true
// (scipy.spatial.distance.cdist with metric 'euclidean'/'sqeuclidean').
NUMPP_API ndarray cdist_euclidean(const ndarray& A, const ndarray& B,
                                  bool squared = false, Backend forced = Backend::Auto);

// Boundary handling for correlate1d (matches scipy.ndimage modes).
enum class FilterMode { Reflect = 0, Constant = 1, Nearest = 2, Mirror = 3, Wrap = 4 };

// Separable 1-D correlation of `input` with `weights` along `axis`
// (scipy.ndimage.correlate1d): out[i] = sum_k weights[k]·input[i + k - (klen/2 +
// origin)] with the boundary resolved per `mode` (Constant uses `cval`).
NUMPP_API ndarray correlate1d(const ndarray& input, const ndarray& weights,
                              int64_t axis = -1, int64_t origin = 0,
                              FilterMode mode = FilterMode::Reflect, double cval = 0.0,
                              Backend forced = Backend::Auto);

}  // namespace numpp
