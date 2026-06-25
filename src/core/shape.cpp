#include "numpp/core/shape.hpp"

#include <algorithm>
#include <string>

namespace numpp {

Strides contiguous_strides(const Shape& shape, int64_t itemsize, Order order) {
  const int64_t n = static_cast<int64_t>(shape.size());
  Strides st(n, 0);
  if (n == 0) return st;
  if (order == Order::C) {
    int64_t acc = itemsize;
    for (int64_t i = n - 1; i >= 0; --i) { st[i] = acc; acc *= shape[i]; }
  } else {
    int64_t acc = itemsize;
    for (int64_t i = 0; i < n; ++i) { st[i] = acc; acc *= shape[i]; }
  }
  return st;
}

int64_t normalize_axis(int64_t axis, int64_t ndim) {
  const int64_t a = axis < 0 ? axis + ndim : axis;
  if (a < 0 || a >= ndim) {
    throw axis_error("axis " + std::to_string(axis) + " is out of bounds for array of dimension " +
                     std::to_string(ndim));
  }
  return a;
}

Shape broadcast_shapes(const Shape& a, const Shape& b) {
  const int64_t na = static_cast<int64_t>(a.size());
  const int64_t nb = static_cast<int64_t>(b.size());
  const int64_t n = std::max(na, nb);
  Shape out(n, 0);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t da = i < n - na ? 1 : a[i - (n - na)];
    const int64_t db = i < n - nb ? 1 : b[i - (n - nb)];
    if (da == db || da == 1 || db == 1) {
      out[i] = std::max(da, db);
    } else {
      throw value_error("operands could not be broadcast together");
    }
  }
  return out;
}

bool is_c_contiguous(const Shape& shape, const Strides& strides, int64_t itemsize) {
  int64_t acc = itemsize;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] == 0) return true;          // empty array is trivially contiguous
    if (shape[i] != 1 && strides[i] != acc) return false;
    acc *= shape[i];
  }
  return true;
}

bool is_f_contiguous(const Shape& shape, const Strides& strides, int64_t itemsize) {
  int64_t acc = itemsize;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) return true;
    if (shape[i] != 1 && strides[i] != acc) return false;
    acc *= shape[i];
  }
  return true;
}

}  // namespace numpp
