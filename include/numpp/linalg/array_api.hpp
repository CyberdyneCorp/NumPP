#pragma once

#include <optional>
#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace linalg {

// Array API 2023 linear-algebra aliases. Thin wrappers over existing ndarray
// shape ops, ufuncs/reductions, and linalg::norm.

// Transpose the last two axes (a.ndim() must be >= 2).
NUMPP_API ndarray matrix_transpose(const ndarray& a);

// Permute axes; equivalent to a.transpose(axes).
NUMPP_API ndarray permute_dims(const ndarray& a, const std::vector<int64_t>& axes);

// numpy.vecdot: sum over `axis` of conj(a) * b (conjugates the first argument).
NUMPP_API ndarray vecdot(const ndarray& a, const ndarray& b, int64_t axis = -1);

// numpy.linalg.vector_norm: 2-norm by default. `axis` nullopt flattens the input.
NUMPP_API ndarray vector_norm(const ndarray& x, std::optional<int64_t> axis = std::nullopt,
                              double ord = 2.0);

// numpy.linalg.matrix_norm: Frobenius by default; reuses linalg::norm.
NUMPP_API ndarray matrix_norm(const ndarray& x, const std::string& ord = "fro");

}  // namespace linalg
}  // namespace numpp
