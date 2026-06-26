#pragma once

#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Assemble blocks from nested lists (numpy.block); 2-level (rows of column-blocks).
NUMPP_API ndarray block(const std::vector<std::vector<ndarray>>& blocks);
// Split along axis 2 (numpy.dsplit); requires ndim >= 3.
NUMPP_API std::vector<ndarray> dsplit(const ndarray& a, int64_t sections);
// Trim leading/trailing zeros from a 1-D array. trim: "f", "b", or "fb".
NUMPP_API ndarray trim_zeros(const ndarray& a, const std::string& trim = "fb");
// Roll `axis` until it lies before `start` (numpy.rollaxis).
NUMPP_API ndarray rollaxis(const ndarray& a, int64_t axis, int64_t start = 0);

}  // namespace numpp
