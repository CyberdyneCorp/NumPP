#include "numpp/backend/lapack_vtable.hpp"

// Default (no-LAPACK) build: no vtable registered, so linalg uses its portable
// pure-C++ kernels. Replaced by lapack_backend.cpp when NUMPP_WITH_LAPACK=ON.

namespace numpp {
const LapackVTable* lapack_vtable() { return nullptr; }
}  // namespace numpp
