#pragma once

#include <optional>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- gather / scatter ----
NUMPP_API ndarray take(const ndarray& a, const ndarray& indices, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray take_along_axis(const ndarray& a, const ndarray& indices, int64_t axis);
NUMPP_API void put(ndarray& a, const ndarray& indices, const ndarray& values);  // in-place, flat (C order)

// ---- extraction ----
NUMPP_API ndarray diagonal(const ndarray& a, int64_t offset = 0, int64_t axis1 = 0, int64_t axis2 = 1);
NUMPP_API ndarray argwhere(const ndarray& a);                                   // (N, ndim) int64
NUMPP_API ndarray compress(const ndarray& condition, const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray extract(const ndarray& condition, const ndarray& a);

// ---- piecewise selection ----
NUMPP_API ndarray choose(const ndarray& indices, const std::vector<ndarray>& choices);
NUMPP_API ndarray select(const std::vector<ndarray>& condlist, const std::vector<ndarray>& choicelist,
                         double default_value = 0.0);

// ---- flat <-> multi index ----
NUMPP_API ndarray ravel_multi_index(const std::vector<ndarray>& multi_index, const Shape& dims);
NUMPP_API std::vector<ndarray> unravel_index(const ndarray& indices, const Shape& dims);

}  // namespace numpp
