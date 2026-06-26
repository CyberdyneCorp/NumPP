#include "numpp/ma/masked_extra.hpp"

#include "numpp/ma/masked.hpp"
#include "numpp/umath/ufunc.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <cmath>
#include <vector>


namespace numpp {

namespace ma {

namespace {

// Build a kBool mask of a.shape() by applying `pred` to each (float64) element.
template <class Pred>
ndarray build_mask(const ndarray& a, Pred pred) {
  ndarray flat = a.astype(kFloat64).ravel(Order::C).ascontiguousarray();
  const int64_t n = flat.size();
  const double* p = flat.typed_data<double>();
  ndarray mask(a.shape(), kBool);
  ndarray mflat = mask.ravel(Order::C);  // contiguous view (mask owns C-order)
  bool* mp = mflat.typed_data<bool>();
  for (int64_t i = 0; i < n; ++i) mp[i] = pred(p[i]);
  return mask;
}

// Elementwise arithmetic: data = op(a.data, b.data); mask = a.mask | b.mask,
// broadcast to the result shape via logical_or.
template <class Op>
MaskedArray elementwise(const MaskedArray& a, const MaskedArray& b, Op op) {
  ndarray data = op(a.data, b.data);
  ndarray mask = ::numpp::logical_or(a.mask, b.mask).astype(kBool);
  return masked_array(data, mask);
}

}  // namespace

MaskedArray add(const MaskedArray& a, const MaskedArray& b) {
  return elementwise(a, b, [](const ndarray& x, const ndarray& y) { return ::numpp::add(x, y); });
}

MaskedArray subtract(const MaskedArray& a, const MaskedArray& b) {
  return elementwise(a, b, [](const ndarray& x, const ndarray& y) { return ::numpp::subtract(x, y); });
}

MaskedArray multiply(const MaskedArray& a, const MaskedArray& b) {
  return elementwise(a, b, [](const ndarray& x, const ndarray& y) { return ::numpp::multiply(x, y); });
}

MaskedArray divide(const MaskedArray& a, const MaskedArray& b) {
  return elementwise(a, b, [](const ndarray& x, const ndarray& y) { return ::numpp::divide(x, y); });
}

MaskedArray masked_less(const ndarray& a, double value) {
  return masked_array(a, build_mask(a, [value](double x) { return x < value; }));
}

MaskedArray masked_less_equal(const ndarray& a, double value) {
  return masked_array(a, build_mask(a, [value](double x) { return x <= value; }));
}

MaskedArray masked_greater_equal(const ndarray& a, double value) {
  return masked_array(a, build_mask(a, [value](double x) { return x >= value; }));
}

MaskedArray masked_not_equal(const ndarray& a, double value) {
  return masked_array(a, build_mask(a, [value](double x) { return x != value; }));
}

MaskedArray masked_inside(const ndarray& a, double v1, double v2) {
  double lo = v1, hi = v2;
  if (lo > hi) { double t = lo; lo = hi; hi = t; }
  return masked_array(a, build_mask(a, [lo, hi](double x) { return x >= lo && x <= hi; }));
}

MaskedArray masked_outside(const ndarray& a, double v1, double v2) {
  double lo = v1, hi = v2;
  if (lo > hi) { double t = lo; lo = hi; hi = t; }
  return masked_array(a, build_mask(a, [lo, hi](double x) { return x < lo || x > hi; }));
}

MaskedArray masked_values(const ndarray& a, double value, double rtol, double atol) {
  const double tol = atol + rtol * std::fabs(value);
  return masked_array(a, build_mask(a, [value, tol](double x) {
    return std::fabs(x - value) <= tol;
  }));
}

ndarray getmask(const MaskedArray& m) { return m.mask; }

ndarray getdata(const MaskedArray& m) { return m.data; }

}  // namespace ma

}  // namespace numpp
