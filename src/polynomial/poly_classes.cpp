#include "numpp/polynomial/poly_classes.hpp"

#include <algorithm>
#include <limits>

#include "numpp/core/creation.hpp"
#include "numpp/linalg/linalg.hpp"

namespace numpp {
namespace polynomial {
namespace detail {

ndarray map_domain(const ndarray& x, const std::array<double, 2>& domain,
                   const std::array<double, 2>& window) {
  ndarray xf = x.astype(kFloat64).ravel().copy();
  const int64_t n = xf.size();
  double* p = n ? xf.typed_data<double>() : nullptr;
  const double dspan = domain[1] - domain[0];
  // Degenerate domain: numpy keeps a unit scale; offset to the window origin.
  const double scl = dspan != 0.0 ? (window[1] - window[0]) / dspan : 1.0;
  const double off = window[0] - scl * domain[0];
  for (int64_t i = 0; i < n; ++i) p[i] = off + scl * p[i];
  return xf;
}

ndarray ortho_lstsq(const ndarray& V, const ndarray& y) {
  return linalg::lstsq(V, y.astype(kFloat64).ravel().copy()).solution;
}

std::array<double, 2> data_range(const ndarray& x) {
  ndarray xf = x.astype(kFloat64).ravel().copy();
  const int64_t n = xf.size();
  if (n == 0) return {-1, 1};
  const double* p = xf.typed_data<double>();
  double lo = p[0], hi = p[0];
  for (int64_t i = 1; i < n; ++i) {
    lo = std::min(lo, p[i]);
    hi = std::max(hi, p[i]);
  }
  return {lo, hi};
}

}  // namespace detail
}  // namespace polynomial
}  // namespace numpp
