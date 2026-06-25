#pragma once

#include <functional>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- coordinate grids ----
// meshgrid: 'xy' (cartesian, default) or 'ij' (matrix) indexing, N inputs.
NUMPP_API std::vector<ndarray> meshgrid(const std::vector<ndarray>& xi, bool indexing_xy = true);
// indices: grid of index arrays, shape (ndim, *shape), int64.
NUMPP_API ndarray indices(const Shape& shape);

// ---- diagonals & triangles ----
NUMPP_API ndarray diag(const ndarray& v, int64_t k = 0);      // 1-D -> matrix; 2-D -> extract diagonal
NUMPP_API ndarray diagflat(const ndarray& v, int64_t k = 0);  // flatten then build matrix
NUMPP_API ndarray tri(int64_t n, int64_t m = -1, int64_t k = 0, DType dtype = kFloat64);
NUMPP_API ndarray tril(const ndarray& m, int64_t k = 0);
NUMPP_API ndarray triu(const ndarray& m, int64_t k = 0);

// ---- generators ----
NUMPP_API ndarray vander(const ndarray& x, int64_t n = -1, bool increasing = false);
NUMPP_API ndarray logspace(double start, double stop, int64_t num = 50, bool endpoint = true,
                           double base = 10.0, DType dtype = kFloat64);
NUMPP_API ndarray geomspace(double start, double stop, int64_t num = 50, bool endpoint = true,
                            DType dtype = kFloat64);
NUMPP_API ndarray fromfunction(const std::function<double(const std::vector<int64_t>&)>& fn,
                               const Shape& shape, DType dtype = kFloat64);

}  // namespace numpp
