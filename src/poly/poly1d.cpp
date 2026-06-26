#include "numpp/poly/poly1d.hpp"

#include "numpp/poly/poly.hpp"
#include "numpp/grids/grids.hpp"
#include "numpp/linalg/linalg.hpp"
#include "numpp/umath/ufunc.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <vector>


namespace numpp {

namespace {

// Normalize coefficients to a contiguous float64 1-D array with leading
// (highest-order) zeros trimmed, keeping at least one coefficient.
ndarray normalize_coeffs(const ndarray& coeffs) {
  ndarray f = coeffs.astype(kFloat64).ravel().ascontiguousarray();
  const int64_t n = f.size();
  const double* p = n ? f.typed_data<double>() : nullptr;
  int64_t start = 0;
  while (start < n - 1 && p[start] == 0.0) ++start;  // keep at least 1
  if (start == 0) return f;
  const int64_t m = n - start;
  ndarray out({m}, kFloat64, Order::C);
  double* o = out.typed_data<double>();
  for (int64_t i = 0; i < m; ++i) o[i] = p[start + i];
  return out;
}

}  // namespace

poly1d::poly1d(const ndarray& coeffs) : coeffs_(normalize_coeffs(coeffs)) {}

ndarray poly1d::coeffs() const { return coeffs_; }

int64_t poly1d::order() const { return coeffs_.size() - 1; }

ndarray poly1d::operator()(const ndarray& x) const { return numpp::polyval(coeffs_, x); }

poly1d poly1d::deriv(int64_t m) const { return poly1d(numpp::polyder(coeffs_, m)); }

poly1d poly1d::integ(int64_t m, double k) const { return poly1d(numpp::polyint(coeffs_, m, k)); }

ndarray poly1d::roots() const { return numpp::roots(coeffs_); }

poly1d poly1d::operator+(const poly1d& o) const { return poly1d(numpp::polyadd(coeffs_, o.coeffs_)); }

poly1d poly1d::operator-(const poly1d& o) const { return poly1d(numpp::polysub(coeffs_, o.coeffs_)); }

poly1d poly1d::operator*(const poly1d& o) const { return poly1d(numpp::polymul(coeffs_, o.coeffs_)); }

ndarray polyfit_weighted(const ndarray& x, const ndarray& y, int64_t deg, const ndarray& w) {
  // Vandermonde matrix, highest-power first columns (decreasing).
  ndarray V = vander(x, deg + 1, false).astype(kFloat64).ascontiguousarray();
  ndarray yf = y.astype(kFloat64).ravel().ascontiguousarray();
  ndarray wf = w.astype(kFloat64).ravel().ascontiguousarray();

  const int64_t rows = V.shape()[0];
  const int64_t cols = V.shape()[1];
  const double* pw = rows ? wf.typed_data<double>() : nullptr;
  const double* py = rows ? yf.typed_data<double>() : nullptr;

  // numpy.polyfit with weights: lhs = V * w[:,None], rhs = y * w
  ndarray Vw({rows, cols}, kFloat64, Order::C);
  double* pv = V.size() ? V.typed_data<double>() : nullptr;
  double* pvw = Vw.size() ? Vw.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) pvw[i * cols + j] = pv[i * cols + j] * pw[i];
  }
  ndarray yw({rows}, kFloat64, Order::C);
  double* pyw = yw.size() ? yw.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < rows; ++i) pyw[i] = py[i] * pw[i];

  linalg::LstsqResult res = linalg::lstsq(Vw, yw);
  return res.solution.astype(kFloat64);
}

}  // namespace numpp
