#pragma once

#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Increasing-power Vandermonde (numpy.polynomial.polynomial.polyvander):
// (len(x), deg+1), column j = x**j.
NUMPP_API ndarray polyvander(const ndarray& x, int64_t deg);
// Companion matrix of a lowest-first power-series (numpy.polynomial.polynomial.polycompanion).
NUMPP_API ndarray polycompanion(const ndarray& c);
// numpy.mask_indices: indices where mask_func(ones(n,n), k) is nonzero.
// mask_func is "tril" or "triu".
NUMPP_API std::vector<ndarray> mask_indices(int64_t n, const std::string& mask_func, int64_t k = 0);

}  // namespace numpp
