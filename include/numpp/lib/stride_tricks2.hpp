#pragma once

#include <functional>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Apply a scalar function elementwise (numpy.vectorize for a 1-arg ufunc-like).
NUMPP_API ndarray vectorize(const std::function<double(double)>& func, const ndarray& a);
// Apply a reduction repeatedly over each axis in `axes`, keeping dims (numpy.apply_over_axes).
NUMPP_API ndarray apply_over_axes(const std::function<ndarray(const ndarray&, int64_t)>& func,
                                  const ndarray& a, const std::vector<int64_t>& axes);

}  // namespace numpp
