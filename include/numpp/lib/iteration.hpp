#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Every multi-index of `shape` in C (row-major) order (numpy.ndindex). An empty
// shape () yields a single empty index; any zero-length axis yields none.
NUMPP_API std::vector<std::vector<int64_t>> ndindex(const Shape& shape);

// (multi-index, value-as-double) pairs in C order for real numeric dtypes
// (numpy.ndenumerate). Throws type_error for complex dtypes.
NUMPP_API std::vector<std::pair<std::vector<int64_t>, double>> ndenumerate(const ndarray& a);

// All elements visited in C order as a 1-D float64 array (numpy.nditer with
// order='C'); honors strides, so transposed / non-contiguous / broadcasted views
// work. Throws type_error for complex dtypes.
NUMPP_API ndarray nditer(const ndarray& a);

}  // namespace numpp
