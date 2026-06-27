#pragma once

#include <cstdint>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Composite trapezoidal integration of `y` along `axis` (the axis is removed
// from the result). `trapezoid` is the current NumPy name; `trapz` is the
// legacy alias. With `dx` the sample spacing is uniform; with `x` the spacing
// is the consecutive difference of `x` along the axis (`x` is either 1-D with
// length matching the axis, or the same shape as `y`). A negative `axis` is
// normalized against `y.ndim()` first.
NUMPP_API ndarray trapezoid(const ndarray& y, double dx = 1.0, int64_t axis = -1);
NUMPP_API ndarray trapezoid(const ndarray& y, const ndarray& x, int64_t axis = -1);
NUMPP_API ndarray trapz(const ndarray& y, double dx = 1.0, int64_t axis = -1);
NUMPP_API ndarray trapz(const ndarray& y, const ndarray& x, int64_t axis = -1);

}  // namespace numpp
