#include "numpp/polynomial/polynomial.hpp"

#include "numpp/core/creation.hpp"   // zeros
#include "numpp/grids/grids.hpp"     // vander
#include "numpp/linalg/linalg.hpp"   // eigvals, lstsq
#include "numpp/umath/ufunc.hpp"     // real

#include <vector>

namespace numpp {
namespace polynomial {
namespace {

std::vector<double> to_vec(const ndarray& a) {
  ndarray f = a.astype(kFloat64).ravel().copy();
  const int64_t n = f.size();
  const double* p = n ? f.typed_data<double>() : nullptr;
  return std::vector<double>(p, p + n);
}
ndarray from_vec(const std::vector<double>& v) {
  ndarray o({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) o.typed_data<double>()[i] = v[i];
  return o;
}

// Evaluate sum_i c[i] * B_i(x) where B follows a 3-term recurrence
//   B_0 = b0, B_1 = b1(x), B_{n+1} = (alpha(n)*x + beta(n)) * B_n + gamma(n) * B_{n-1}.
ndarray basis_eval(const ndarray& x, const std::vector<double>& c, double b0,
                   double a1, double b1c,
                   double (*alpha)(int64_t), double (*beta)(int64_t), double (*gamma)(int64_t)) {
  ndarray xf = x.astype(kFloat64).ascontiguousarray();
  ndarray out(xf.shape(), kFloat64, Order::C);
  const double* px = xf.size() ? xf.typed_data<double>() : nullptr;
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  const int64_t deg = static_cast<int64_t>(c.size());
  for (int64_t i = 0; i < xf.size(); ++i) {
    const double xv = px[i];
    double acc = 0.0;
    if (deg > 0) {
      double bm1 = b0;              // B_0
      double bn = a1 * xv + b1c;    // B_1
      acc = c[0] * bm1;
      if (deg > 1) acc += c[1] * bn;
      for (int64_t n = 1; n + 1 < deg; ++n) {
        const double bnp = (alpha(n) * xv + beta(n)) * bn + gamma(n) * bm1;
        acc += c[n + 1] * bnp;
        bm1 = bn;
        bn = bnp;
      }
    }
    o[i] = acc;
  }
  return out;
}

}  // namespace

ndarray polyval(const ndarray& x, const ndarray& c) {
  std::vector<double> cv = to_vec(c);
  ndarray xf = x.astype(kFloat64).ascontiguousarray();
  ndarray out(xf.shape(), kFloat64, Order::C);
  const double* px = xf.size() ? xf.typed_data<double>() : nullptr;
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < xf.size(); ++i) {
    double r = 0.0;
    for (int64_t k = static_cast<int64_t>(cv.size()) - 1; k >= 0; --k) r = r * px[i] + cv[k];
    o[i] = r;
  }
  return out;
}

ndarray polyadd(const ndarray& a, const ndarray& b) {
  std::vector<double> av = to_vec(a), bv = to_vec(b);
  std::vector<double> out(std::max(av.size(), bv.size()), 0.0);
  for (size_t i = 0; i < av.size(); ++i) out[i] += av[i];
  for (size_t i = 0; i < bv.size(); ++i) out[i] += bv[i];
  return from_vec(out);
}
ndarray polysub(const ndarray& a, const ndarray& b) {
  std::vector<double> av = to_vec(a), bv = to_vec(b);
  std::vector<double> out(std::max(av.size(), bv.size()), 0.0);
  for (size_t i = 0; i < av.size(); ++i) out[i] += av[i];
  for (size_t i = 0; i < bv.size(); ++i) out[i] -= bv[i];
  return from_vec(out);
}
ndarray polymul(const ndarray& a, const ndarray& b) {
  std::vector<double> av = to_vec(a), bv = to_vec(b);
  if (av.empty() || bv.empty()) return from_vec({0.0});
  std::vector<double> out(av.size() + bv.size() - 1, 0.0);
  for (size_t i = 0; i < av.size(); ++i)
    for (size_t j = 0; j < bv.size(); ++j) out[i + j] += av[i] * bv[j];
  return from_vec(out);
}

ndarray polyder(const ndarray& c, int64_t m) {
  std::vector<double> v = to_vec(c);
  for (int64_t s = 0; s < m; ++s) {
    if (v.size() <= 1) { v = {0.0}; break; }
    std::vector<double> d(v.size() - 1);
    for (size_t i = 1; i < v.size(); ++i) d[i - 1] = v[i] * static_cast<double>(i);
    v = d;
  }
  return from_vec(v);
}

ndarray polyint(const ndarray& c, int64_t m, double k) {
  std::vector<double> v = to_vec(c);
  for (int64_t s = 0; s < m; ++s) {
    std::vector<double> ic(v.size() + 1, 0.0);
    ic[0] = k;
    for (size_t i = 0; i < v.size(); ++i) ic[i + 1] = v[i] / static_cast<double>(i + 1);
    v = ic;
  }
  return from_vec(v);
}

ndarray polyroots(const ndarray& c) {
  std::vector<double> v = to_vec(c);
  while (v.size() > 1 && v.back() == 0.0) v.pop_back();  // strip high-order zeros
  const int64_t N = static_cast<int64_t>(v.size()) - 1;
  if (N < 1) return from_vec({});
  // companion matrix for lowest-first coefficients
  ndarray comp = zeros({N, N}, kFloat64);
  for (int64_t i = 0; i < N; ++i) comp.set_item<double>({i, N - 1}, -v[i] / v[N]);
  for (int64_t i = 1; i < N; ++i) comp.set_item<double>({i, i - 1}, 1.0);
  ndarray ev = real(linalg::eigvals(comp)).astype(kFloat64);
  return ev;
}

ndarray polyfit(const ndarray& x, const ndarray& y, int64_t deg) {
  ndarray V = vander(x, deg + 1, true);  // increasing powers -> lowest-first coeffs
  linalg::LstsqResult res = linalg::lstsq(V, y);
  return res.solution.astype(kFloat64);
}

// recurrence coefficient helpers
namespace {
double cheb_alpha(int64_t) { return 2.0; }
double cheb_beta(int64_t) { return 0.0; }
double cheb_gamma(int64_t) { return -1.0; }
double leg_alpha(int64_t n) { return static_cast<double>(2 * n + 1) / static_cast<double>(n + 1); }
double leg_beta(int64_t) { return 0.0; }
double leg_gamma(int64_t n) { return -static_cast<double>(n) / static_cast<double>(n + 1); }
double herm_alpha(int64_t) { return 2.0; }
double herm_beta(int64_t) { return 0.0; }
double herm_gamma(int64_t n) { return -2.0 * static_cast<double>(n); }
double herme_alpha(int64_t) { return 1.0; }
double herme_beta(int64_t) { return 0.0; }
double herme_gamma(int64_t n) { return -static_cast<double>(n); }
double lag_alpha(int64_t n) { return -1.0 / static_cast<double>(n + 1); }
double lag_beta(int64_t n) { return static_cast<double>(2 * n + 1) / static_cast<double>(n + 1); }
double lag_gamma(int64_t n) { return -static_cast<double>(n) / static_cast<double>(n + 1); }
}  // namespace

ndarray chebval(const ndarray& x, const ndarray& c) {
  return basis_eval(x, to_vec(c), 1.0, 1.0, 0.0, cheb_alpha, cheb_beta, cheb_gamma);
}
ndarray legval(const ndarray& x, const ndarray& c) {
  return basis_eval(x, to_vec(c), 1.0, 1.0, 0.0, leg_alpha, leg_beta, leg_gamma);
}
ndarray hermval(const ndarray& x, const ndarray& c) {
  return basis_eval(x, to_vec(c), 1.0, 2.0, 0.0, herm_alpha, herm_beta, herm_gamma);
}
ndarray hermeval(const ndarray& x, const ndarray& c) {
  return basis_eval(x, to_vec(c), 1.0, 1.0, 0.0, herme_alpha, herme_beta, herme_gamma);
}
ndarray lagval(const ndarray& x, const ndarray& c) {
  return basis_eval(x, to_vec(c), 1.0, -1.0, 1.0, lag_alpha, lag_beta, lag_gamma);
}

// ---- Polynomial class ----
Polynomial::Polynomial(const ndarray& coef) : coef_(to_vec(coef)) {}
Polynomial::Polynomial(std::vector<double> coef) : coef_(std::move(coef)) {}
ndarray Polynomial::coef() const { return from_vec(coef_); }
ndarray Polynomial::operator()(const ndarray& x) const { return polyval(x, from_vec(coef_)); }
Polynomial Polynomial::deriv(int64_t m) const { return Polynomial(to_vec(polyder(from_vec(coef_), m))); }
Polynomial Polynomial::integ(int64_t m, double k) const { return Polynomial(to_vec(polyint(from_vec(coef_), m, k))); }
ndarray Polynomial::roots() const { return polyroots(from_vec(coef_)); }
Polynomial Polynomial::operator+(const Polynomial& o) const { return Polynomial(to_vec(polyadd(from_vec(coef_), from_vec(o.coef_)))); }
Polynomial Polynomial::operator-(const Polynomial& o) const { return Polynomial(to_vec(polysub(from_vec(coef_), from_vec(o.coef_)))); }
Polynomial Polynomial::operator*(const Polynomial& o) const { return Polynomial(to_vec(polymul(from_vec(coef_), from_vec(o.coef_)))); }
Polynomial Polynomial::fit(const ndarray& x, const ndarray& y, int64_t deg) { return Polynomial(to_vec(polyfit(x, y, deg))); }

}  // namespace polynomial
}  // namespace numpp
