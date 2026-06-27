#include "numpp/mathx/closeto.hpp"

#include "numpp/core/shape.hpp"  // broadcast_shapes

#include <cmath>

namespace numpp {
namespace {

// Per-element closeness, mirroring numpy.isclose semantics for inf/nan.
bool close_scalar(double x, double y, double rtol, double atol, bool equal_nan) {
  if (std::isnan(x) || std::isnan(y)) return equal_nan && std::isnan(x) && std::isnan(y);
  if (std::isinf(x) || std::isinf(y)) return x == y;  // matching infinities only
  return std::abs(x - y) <= atol + rtol * std::abs(y);
}

}  // namespace

ndarray isclose(const ndarray& a, const ndarray& b, double rtol, double atol, bool equal_nan) {
  const Shape sh = broadcast_shapes(a.shape(), b.shape());
  ndarray af = a.astype(kFloat64).broadcast_to(sh).copy();
  ndarray bf = b.astype(kFloat64).broadcast_to(sh).copy();
  ndarray out(sh, kBool, Order::C);
  const double* pa = af.size() ? af.typed_data<double>() : nullptr;
  const double* pb = bf.size() ? bf.typed_data<double>() : nullptr;
  bool* o = out.size() ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < out.size(); ++i) o[i] = close_scalar(pa[i], pb[i], rtol, atol, equal_nan);
  return out;
}

ndarray isposinf(const ndarray& a) {
  ndarray x = a.astype(kFloat64).ascontiguousarray();
  ndarray out(x.shape(), kBool, Order::C);
  const double* p = x.size() ? x.typed_data<double>() : nullptr;
  bool* o = out.size() ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < x.size(); ++i) o[i] = std::isinf(p[i]) && p[i] > 0.0;
  return out;
}

ndarray isneginf(const ndarray& a) {
  ndarray x = a.astype(kFloat64).ascontiguousarray();
  ndarray out(x.shape(), kBool, Order::C);
  const double* p = x.size() ? x.typed_data<double>() : nullptr;
  bool* o = out.size() ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < x.size(); ++i) o[i] = std::isinf(p[i]) && p[i] < 0.0;
  return out;
}

}  // namespace numpp
