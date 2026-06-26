#include "numpp/polynomial/orthopoly.hpp"

#include "numpp/core/creation.hpp"
#include "numpp/linalg/linalg.hpp"
#include "numpp/umath/ufunc.hpp"
#include "numpp/core/error.hpp"
#include <cmath>
#include <vector>
#include <algorithm>


namespace numpp {

namespace polynomial {
namespace {

std::vector<double> op_to_vec(const ndarray& a) {
  ndarray f = a.astype(kFloat64).ravel().copy();
  const int64_t n = f.size();
  const double* p = n ? f.typed_data<double>() : nullptr;
  return std::vector<double>(p, p + n);
}
ndarray op_from_vec(const std::vector<double>& v) {
  ndarray o({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) o.typed_data<double>()[i] = v[i];
  return o;
}
// strip trailing (highest-degree) zeros
std::vector<double> op_trim(std::vector<double> v) {
  while (v.size() > 1 && v.back() == 0.0) v.pop_back();
  return v;
}

// Build a (deg+1)-column Vandermonde matrix using the basis 3-term recurrence
//   col0 = 1, col1 = a1*x + b1c,
//   col_{j+1} = (alpha(j)*x + beta(j))*col_j + gamma(j)*col_{j-1}.
ndarray op_vander(const ndarray& x, int64_t deg, double a1, double b1c,
                  double (*alpha)(int64_t), double (*beta)(int64_t), double (*gamma)(int64_t)) {
  if (deg < 0) value_error("deg must be non-negative");
  ndarray xf = x.astype(kFloat64).ravel().ascontiguousarray();
  const int64_t m = xf.size();
  ndarray out({m, deg + 1}, kFloat64, Order::C);
  const double* px = m ? xf.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < m; ++i) out.set_item<double>({i, 0}, 1.0);
  if (deg >= 1)
    for (int64_t i = 0; i < m; ++i) out.set_item<double>({i, 1}, a1 * px[i] + b1c);
  for (int64_t j = 1; j < deg; ++j) {
    for (int64_t i = 0; i < m; ++i) {
      const double v = (alpha(j) * px[i] + beta(j)) * out.item<double>({i, j}) +
                       gamma(j) * out.item<double>({i, j - 1});
      out.set_item<double>({i, j + 1}, v);
    }
  }
  return out;
}

// Assemble an n x n companion matrix from a main diagonal, a symmetric
// off-diagonal, and an additive contribution to the last column. The last
// column adjustment is applied last so it accumulates on top of any
// off-diagonal entry that already lives there (matching numpy).
ndarray op_companion(int64_t n, const std::vector<double>& diag,
                     const std::vector<double>& off, const std::vector<double>& last) {
  ndarray m = zeros({n, n}, kFloat64);
  for (int64_t i = 0; i < n && i < static_cast<int64_t>(diag.size()); ++i)
    m.set_item<double>({i, i}, diag[i]);
  for (int64_t i = 0; i + 1 < n; ++i) {
    m.set_item<double>({i, i + 1}, off[i]);
    m.set_item<double>({i + 1, i}, off[i]);
  }
  for (int64_t i = 0; i < n; ++i)
    m.set_item<double>({i, n - 1}, m.item<double>({i, n - 1}) + last[i]);
  return m;
}

// Real parts of the companion eigenvalues, sorted ascending.
ndarray op_roots_of(const ndarray& comp) {
  ndarray ev = real(linalg::eigvals(comp)).astype(kFloat64).ravel().copy();
  const int64_t n = ev.size();
  double* p = n ? ev.typed_data<double>() : nullptr;
  std::sort(p, p + n);
  return ev;
}

// Cumulative-product scaling vector (reversed) shared by the two Hermite
// companion matrices. two==true -> physicists', false -> probabilists'.
std::vector<double> op_herm_scl(int64_t n, bool two) {
  std::vector<double> base(n);
  base[0] = 1.0;
  for (int64_t k = 1; k < n; ++k)
    base[k] = 1.0 / std::sqrt((two ? 2.0 : 1.0) * static_cast<double>(n - k));
  for (int64_t k = 1; k < n; ++k) base[k] *= base[k - 1];  // accumulate
  std::reverse(base.begin(), base.end());
  return base;
}

// recurrence coefficient helpers
double op_cheb_alpha(int64_t) { return 2.0; }
double op_cheb_beta(int64_t) { return 0.0; }
double op_cheb_gamma(int64_t) { return -1.0; }
double op_leg_alpha(int64_t n) { return static_cast<double>(2 * n + 1) / static_cast<double>(n + 1); }
double op_leg_beta(int64_t) { return 0.0; }
double op_leg_gamma(int64_t n) { return -static_cast<double>(n) / static_cast<double>(n + 1); }
double op_herm_alpha(int64_t) { return 2.0; }
double op_herm_beta(int64_t) { return 0.0; }
double op_herm_gamma(int64_t n) { return -2.0 * static_cast<double>(n); }
double op_herme_alpha(int64_t) { return 1.0; }
double op_herme_beta(int64_t) { return 0.0; }
double op_herme_gamma(int64_t n) { return -static_cast<double>(n); }
double op_lag_alpha(int64_t n) { return -1.0 / static_cast<double>(n + 1); }
double op_lag_beta(int64_t n) { return static_cast<double>(2 * n + 1) / static_cast<double>(n + 1); }
double op_lag_gamma(int64_t n) { return -static_cast<double>(n) / static_cast<double>(n + 1); }

}  // namespace

// ---- Vandermonde matrices ----
ndarray chebvander(const ndarray& x, int64_t deg) {
  return op_vander(x, deg, 1.0, 0.0, op_cheb_alpha, op_cheb_beta, op_cheb_gamma);
}
ndarray legvander(const ndarray& x, int64_t deg) {
  return op_vander(x, deg, 1.0, 0.0, op_leg_alpha, op_leg_beta, op_leg_gamma);
}
ndarray hermvander(const ndarray& x, int64_t deg) {
  return op_vander(x, deg, 2.0, 0.0, op_herm_alpha, op_herm_beta, op_herm_gamma);
}
ndarray hermevander(const ndarray& x, int64_t deg) {
  return op_vander(x, deg, 1.0, 0.0, op_herme_alpha, op_herme_beta, op_herme_gamma);
}
ndarray lagvander(const ndarray& x, int64_t deg) {
  return op_vander(x, deg, -1.0, 1.0, op_lag_alpha, op_lag_beta, op_lag_gamma);
}

// ---- Roots via basis companion matrices (numpy <basis>companion) ----
ndarray chebroots(const ndarray& c) {
  std::vector<double> v = op_trim(op_to_vec(c));
  const int64_t n = static_cast<int64_t>(v.size()) - 1;
  if (n < 1) return op_from_vec({});
  if (n == 1) return op_from_vec({-v[0] / v[1]});
  std::vector<double> off(n - 1), last(n);
  for (int64_t k = 0; k < n - 1; ++k) off[k] = (k == 0) ? std::sqrt(0.5) : 0.5;
  for (int64_t i = 0; i < n; ++i) {
    const double s = (i == 0) ? std::sqrt(0.5) : 0.5;
    last[i] = -(v[i] / v[n]) * s;
  }
  return op_roots_of(op_companion(n, {}, off, last));
}

ndarray legroots(const ndarray& c) {
  std::vector<double> v = op_trim(op_to_vec(c));
  const int64_t n = static_cast<int64_t>(v.size()) - 1;
  if (n < 1) return op_from_vec({});
  if (n == 1) return op_from_vec({-v[0] / v[1]});
  std::vector<double> off(n - 1), last(n);
  for (int64_t k = 0; k < n - 1; ++k)
    off[k] = static_cast<double>(k + 1) /
             std::sqrt(static_cast<double>((2 * k + 1) * (2 * k + 3)));
  for (int64_t i = 0; i < n; ++i) {
    const double f = static_cast<double>(n) /
                     (std::sqrt(static_cast<double>(2 * i + 1)) *
                      std::sqrt(static_cast<double>(2 * n - 1)));
    last[i] = -(v[i] / v[n]) * f;
  }
  return op_roots_of(op_companion(n, {}, off, last));
}

ndarray hermroots(const ndarray& c) {
  std::vector<double> v = op_trim(op_to_vec(c));
  const int64_t n = static_cast<int64_t>(v.size()) - 1;
  if (n < 1) return op_from_vec({});
  if (n == 1) return op_from_vec({-0.5 * v[0] / v[1]});
  std::vector<double> off(n - 1), last(n);
  for (int64_t k = 0; k < n - 1; ++k) off[k] = std::sqrt(0.5 * static_cast<double>(k + 1));
  const std::vector<double> scl = op_herm_scl(n, true);
  for (int64_t i = 0; i < n; ++i) last[i] = -scl[i] * v[i] / v[n] * 0.5;
  return op_roots_of(op_companion(n, {}, off, last));
}

ndarray hermeroots(const ndarray& c) {
  std::vector<double> v = op_trim(op_to_vec(c));
  const int64_t n = static_cast<int64_t>(v.size()) - 1;
  if (n < 1) return op_from_vec({});
  if (n == 1) return op_from_vec({-v[0] / v[1]});
  std::vector<double> off(n - 1), last(n);
  for (int64_t k = 0; k < n - 1; ++k) off[k] = std::sqrt(static_cast<double>(k + 1));
  const std::vector<double> scl = op_herm_scl(n, false);
  for (int64_t i = 0; i < n; ++i) last[i] = -scl[i] * v[i] / v[n];
  return op_roots_of(op_companion(n, {}, off, last));
}

ndarray lagroots(const ndarray& c) {
  std::vector<double> v = op_trim(op_to_vec(c));
  const int64_t n = static_cast<int64_t>(v.size()) - 1;
  if (n < 1) return op_from_vec({});
  if (n == 1) return op_from_vec({1.0 + v[0] / v[1]});
  std::vector<double> diag(n), off(n - 1), last(n);
  for (int64_t k = 0; k < n; ++k) diag[k] = static_cast<double>(2 * k + 1);
  for (int64_t k = 0; k < n - 1; ++k) off[k] = -static_cast<double>(k + 1);
  for (int64_t i = 0; i < n; ++i) last[i] = (v[i] / v[n]) * static_cast<double>(n);
  return op_roots_of(op_companion(n, diag, off, last));
}

}  // namespace polynomial

}  // namespace numpp
