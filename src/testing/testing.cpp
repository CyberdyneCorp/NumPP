#include "numpp/testing/testing.hpp"

#include "numpp/umath/ufunc.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/core/error.hpp"
#include <cmath>
#include <vector>


namespace numpp {

namespace testing {

namespace {

// Flatten to a contiguous float64 view we can read with typed_data<double>().
inline ndarray as_flat_f64(const ndarray& a) {
  return a.astype(kFloat64).ascontiguousarray();
}

// True when both shapes are identical.
inline bool same_shape(const Shape& a, const Shape& b) {
  return a == b;
}

}  // namespace

bool array_equal(const ndarray& a, const ndarray& b) {
  if (!same_shape(a.shape(), b.shape())) return false;
  const ndarray fa = as_flat_f64(a);
  const ndarray fb = as_flat_f64(b);
  const int64_t n = fa.size();
  const double* pa = fa.typed_data<double>();
  const double* pb = fb.typed_data<double>();
  for (int64_t i = 0; i < n; ++i) {
    if (pa[i] != pb[i]) return false;
  }
  return true;
}

bool array_equiv(const ndarray& a, const ndarray& b) {
  Shape bshape;
  try {
    bshape = broadcast_shapes(a.shape(), b.shape());
  } catch (const value_error&) {
    return false;
  }
  const ndarray fa = as_flat_f64(a.broadcast_to(bshape));
  const ndarray fb = as_flat_f64(b.broadcast_to(bshape));
  const int64_t n = fa.size();
  const double* pa = fa.typed_data<double>();
  const double* pb = fb.typed_data<double>();
  for (int64_t i = 0; i < n; ++i) {
    if (pa[i] != pb[i]) return false;
  }
  return true;
}

void assert_array_equal(const ndarray& a, const ndarray& b) {
  if (!testing::array_equal(a, b)) throw value_error("arrays are not equal");
}

void assert_allclose(const ndarray& a, const ndarray& b, double rtol, double atol) {
  const Shape bshape = broadcast_shapes(a.shape(), b.shape());
  const ndarray fa = as_flat_f64(a.broadcast_to(bshape));
  const ndarray fb = as_flat_f64(b.broadcast_to(bshape));
  const int64_t n = fa.size();
  const double* pa = fa.typed_data<double>();
  const double* pb = fb.typed_data<double>();
  for (int64_t i = 0; i < n; ++i) {
    const double diff = pa[i] - pb[i];
    const double ad = diff < 0 ? -diff : diff;
    const double ab = pb[i] < 0 ? -pb[i] : pb[i];
    if (!(ad <= atol + rtol * ab)) throw value_error("arrays are not close");
  }
}

void assert_array_almost_equal(const ndarray& a, const ndarray& b, int decimal) {
  const Shape bshape = broadcast_shapes(a.shape(), b.shape());
  const ndarray fa = as_flat_f64(a.broadcast_to(bshape));
  const ndarray fb = as_flat_f64(b.broadcast_to(bshape));
  const int64_t n = fa.size();
  const double* pa = fa.typed_data<double>();
  const double* pb = fb.typed_data<double>();
  double tol = 1.5;
  for (int d = 0; d < decimal; ++d) tol *= 0.1;
  for (int64_t i = 0; i < n; ++i) {
    const double diff = pa[i] - pb[i];
    const double ad = diff < 0 ? -diff : diff;
    if (!(ad < tol)) throw value_error("arrays are not almost equal");
  }
}

void assert_array_less(const ndarray& a, const ndarray& b) {
  const Shape bshape = broadcast_shapes(a.shape(), b.shape());
  const ndarray fa = as_flat_f64(a.broadcast_to(bshape));
  const ndarray fb = as_flat_f64(b.broadcast_to(bshape));
  const int64_t n = fa.size();
  const double* pa = fa.typed_data<double>();
  const double* pb = fb.typed_data<double>();
  for (int64_t i = 0; i < n; ++i) {
    if (!(pa[i] < pb[i])) throw value_error("arrays are not less-ordered");
  }
}

}  // namespace testing

}  // namespace numpp
