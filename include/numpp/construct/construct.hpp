#pragma once

#include <array>
#include <string>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Build a 1-D array from a flat value list (numpy.fromiter).
NUMPP_API ndarray fromiter(const std::vector<double>& data, DType dtype = kFloat64);
// Reinterpret raw bytes as an array (numpy.frombuffer). count<0 => use all bytes.
NUMPP_API ndarray frombuffer(const std::string& buffer, DType dtype, int64_t count = -1);
// Broadcast several arrays against one another, returning views of the common shape.
NUMPP_API std::vector<ndarray> broadcast_arrays(const std::vector<ndarray>& arrays);
// meshgrid with sparse outputs (numpy.meshgrid(..., sparse=True)): each output keeps
// length 1 on every axis except its own.
NUMPP_API std::vector<ndarray> meshgrid_sparse(const std::vector<ndarray>& xi, bool indexing_xy = true);
// Dense (mgrid) / open (ogrid) coordinate grids. Each range is {start, stop, step}
// (half-open, like numpy.arange). mgrid returns full N-D grids; ogrid returns
// 1-D-per-axis open grids.
NUMPP_API std::vector<ndarray> mgrid(const std::vector<std::array<double, 3>>& ranges);
NUMPP_API std::vector<ndarray> ogrid(const std::vector<std::array<double, 3>>& ranges);

}  // namespace numpp
