#pragma once

#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// General Einstein summation, e.g. "ij,jk->ik", "ii->i", "ij->ji", "i,i->".
NUMPP_API ndarray einsum(const std::string& subscripts, const std::vector<ndarray>& operands);

// Tensor dot product over the given axes (or last `n` of a with first `n` of b).
NUMPP_API ndarray tensordot(const ndarray& a, const ndarray& b, int64_t n = 2);
NUMPP_API ndarray tensordot(const ndarray& a, const ndarray& b,
                            const std::vector<int64_t>& axes_a, const std::vector<int64_t>& axes_b);

// Cross product over the last axis (length 2 or 3).
NUMPP_API ndarray cross(const ndarray& a, const ndarray& b);

// 2-norm condition number (largest/smallest singular value).
NUMPP_API ndarray cond(const ndarray& a);

// Chained matrix product a0 @ a1 @ ... (left to right).
NUMPP_API ndarray multi_dot(const std::vector<ndarray>& arrays);

}  // namespace numpp
