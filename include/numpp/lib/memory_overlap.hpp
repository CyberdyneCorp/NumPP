#pragma once

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Bounds-based memory-overlap checks (numpy.may_share_memory / numpy.shares_memory).
//
// Both return false when the two arrays reference different underlying Buffers.
// When they share a Buffer, each array's elements occupy an inclusive byte
// interval within that Buffer; the functions report whether those intervals
// intersect. This is exact for may_share_memory and a faithful (conservative)
// bounds check for shares_memory.
NUMPP_API bool may_share_memory(const ndarray& a, const ndarray& b);
NUMPP_API bool shares_memory(const ndarray& a, const ndarray& b);

}  // namespace numpp
