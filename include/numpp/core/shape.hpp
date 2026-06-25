#pragma once

#include <cstdint>
#include <vector>

#include "numpp/core/error.hpp"
#include "numpp/export.hpp"

namespace numpp {

using Shape = std::vector<int64_t>;
using Strides = std::vector<int64_t>;

enum class Order { C, F };

// Number of elements implied by a shape (empty shape -> scalar, size 1).
inline int64_t shape_size(const Shape& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// Row-major (C) or column-major (F) contiguous byte strides for shape+itemsize.
NUMPP_API Strides contiguous_strides(const Shape& shape, int64_t itemsize, Order order);

// Normalize a possibly-negative axis into [0, ndim); throws axis_error otherwise.
NUMPP_API int64_t normalize_axis(int64_t axis, int64_t ndim);

// NumPy broadcasting of two shapes; throws value_error when incompatible.
NUMPP_API Shape broadcast_shapes(const Shape& a, const Shape& b);

// True if shape+strides describe a C- (resp. F-) contiguous layout of itemsize.
NUMPP_API bool is_c_contiguous(const Shape& shape, const Strides& strides, int64_t itemsize);
NUMPP_API bool is_f_contiguous(const Shape& shape, const Strides& strides, int64_t itemsize);

}  // namespace numpp
