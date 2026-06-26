#pragma once

#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Boolean-mask get/set (a[mask], a[mask] = values). mask broadcasts to a.shape.
NUMPP_API ndarray boolean_index(const ndarray& a, const ndarray& mask);
NUMPP_API void boolean_assign(ndarray& a, const ndarray& mask, const ndarray& values);
// Integer-array fancy get/set over the flattened array (a.flat[indices]).
NUMPP_API ndarray fancy_index(const ndarray& a, const ndarray& indices);
NUMPP_API void fancy_assign(ndarray& a, const ndarray& indices, const ndarray& values);

// Scatter values into a along an axis at per-slice indices (numpy.put_along_axis).
NUMPP_API void put_along_axis(ndarray& a, const ndarray& indices, const ndarray& values, int64_t axis);
// Set a[mask] = vals cycling through vals (numpy.place).
NUMPP_API void place(ndarray& a, const ndarray& mask, const ndarray& vals);
// Open mesh from index sequences (numpy.ix_).
NUMPP_API std::vector<ndarray> ix_(const std::vector<ndarray>& seqs);

// Diagonal / triangle index helpers.
NUMPP_API void fill_diagonal(ndarray& a, double val);
NUMPP_API std::vector<ndarray> diag_indices(int64_t n, int64_t ndim = 2);
NUMPP_API std::vector<ndarray> tril_indices(int64_t n, int64_t k = 0, int64_t m = -1);
NUMPP_API std::vector<ndarray> triu_indices(int64_t n, int64_t k = 0, int64_t m = -1);

}  // namespace numpp
