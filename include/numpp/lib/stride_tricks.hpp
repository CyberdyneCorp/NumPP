#pragma once

#include <functional>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// numpy.lib.stride_tricks.sliding_window_view: windows of window_shape over the
// trailing axes; output appends the window dimensions.
NUMPP_API ndarray sliding_window_view(const ndarray& a, const std::vector<int64_t>& window_shape);
// numpy.lib.stride_tricks.as_strided: a view with explicit shape and byte strides.
NUMPP_API ndarray as_strided(const ndarray& a, const Shape& shape, const std::vector<int64_t>& strides_bytes);

// numpy.piecewise: x where condlist[i] selects funclist[i] (default 0 elsewhere).
NUMPP_API ndarray piecewise(const ndarray& x, const std::vector<ndarray>& condlist,
                            const std::vector<std::function<double(double)>>& funclist);
// numpy.apply_along_axis: apply a 1-D->1-D function along an axis.
NUMPP_API ndarray apply_along_axis(const std::function<ndarray(const ndarray&)>& func, int64_t axis,
                                   const ndarray& a);

}  // namespace numpp
