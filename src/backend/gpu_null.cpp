#include "numpp/backend/gpu_vtable.hpp"

// Default build: no GPU backend registered, so the ufunc layer always uses its
// portable CPU kernels. Replaced by refgpu_backend.cpp when NUMPP_WITH_REFGPU=ON.

namespace numpp {
const GpuVTable* gpu_vtable() { return nullptr; }
}  // namespace numpp
