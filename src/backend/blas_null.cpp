#include "numpp/backend/blas_vtable.hpp"

// Default (no-BLAS) build: no vtable registered. This is the portable baseline
// compiled for iOS/Android. Replaced by blas_backend.cpp when NUMPP_WITH_BLAS=ON.

namespace numpp {
const BlasVTable* blas_vtable() { return nullptr; }
}  // namespace numpp
