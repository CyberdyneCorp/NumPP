#include "numpp/stats/integrate.hpp"

#include <vector>

#include "numpp/core/error.hpp"

namespace numpp {
namespace {

int64_t norm_axis(int64_t axis, int64_t ndim) {
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw value_error("trapezoid: axis out of bounds");
  return axis;
}

// y shape with `axis` removed (the integrated dimension collapses away).
Shape reduced_shape(const Shape& yshape, int64_t axis) {
  Shape out;
  for (int64_t i = 0; i < static_cast<int64_t>(yshape.size()); ++i)
    if (i != axis) out.push_back(yshape[i]);
  return out;
}

// Advance a C-order multi-index in place; bounded by `shape`.
void increment_index(std::vector<int64_t>& idx, const Shape& shape) {
  for (int64_t ax = static_cast<int64_t>(idx.size()) - 1; ax >= 0; --ax) {
    if (++idx[ax] < shape[ax]) return;
    idx[ax] = 0;
  }
}

// Expand a reduced (output) index into a full y-index, leaving `axis` at 0.
void expand_index(const std::vector<int64_t>& oidx, int64_t axis, int64_t ndim,
                  std::vector<int64_t>& full) {
  for (int64_t i = 0, j = 0; i < ndim; ++i) full[i] = (i == axis) ? 0 : oidx[j++];
}

// Spacing between sample k and k+1 along `axis` for one integration line.
// `full` already carries the non-axis coordinates of the current line.
double spacing(double dx, const ndarray* x, bool x_is_1d, int64_t axis,
               std::vector<int64_t>& full, int64_t k) {
  if (!x) return dx;
  if (x_is_1d) return x->item<double>({k + 1}) - x->item<double>({k});
  full[axis] = k + 1;
  const double hi = x->item<double>(full);
  full[axis] = k;
  const double lo = x->item<double>(full);
  return hi - lo;
}

ndarray trapezoid_core(const ndarray& yin, double dx, const ndarray* xin, int64_t axis) {
  const ndarray y = yin.astype(kFloat64);
  const int64_t ndim = y.ndim();
  axis = norm_axis(axis, ndim);
  const int64_t n = y.shape()[axis];

  ndarray x;
  bool x_is_1d = false;
  if (xin) {
    x = xin->astype(kFloat64).ascontiguousarray();
    x_is_1d = x.ndim() == 1;
    if (x_is_1d) {
      if (x.shape()[0] != n) throw value_error("trapezoid: x length must match the axis");
    } else if (x.shape() != y.shape()) {
      throw value_error("trapezoid: x must be 1-D or the same shape as y");
    }
  }

  const Shape osh = reduced_shape(y.shape(), axis);
  ndarray out(osh, kFloat64, Order::C);
  if (out.size() == 0) return out;

  std::vector<int64_t> oidx(osh.size(), 0);
  std::vector<int64_t> full(ndim, 0);
  double* o = out.typed_data<double>();
  for (int64_t lin = 0; lin < out.size(); ++lin) {
    expand_index(oidx, axis, ndim, full);
    double acc = 0.0;
    for (int64_t k = 0; k + 1 < n; ++k) {
      const double d = spacing(dx, xin ? &x : nullptr, x_is_1d, axis, full, k);
      full[axis] = k;
      const double yk = y.item<double>(full);
      full[axis] = k + 1;
      const double yk1 = y.item<double>(full);
      acc += d * (yk + yk1) * 0.5;
    }
    o[lin] = acc;
    increment_index(oidx, osh);
  }
  return out;
}

}  // namespace

ndarray trapezoid(const ndarray& y, double dx, int64_t axis) {
  return trapezoid_core(y, dx, nullptr, axis);
}
ndarray trapezoid(const ndarray& y, const ndarray& x, int64_t axis) {
  return trapezoid_core(y, 1.0, &x, axis);
}
ndarray trapz(const ndarray& y, double dx, int64_t axis) {
  return trapezoid_core(y, dx, nullptr, axis);
}
ndarray trapz(const ndarray& y, const ndarray& x, int64_t axis) {
  return trapezoid_core(y, 1.0, &x, axis);
}

}  // namespace numpp
