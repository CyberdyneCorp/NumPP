#include "numpp/poly/poly.hpp"

#include "numpp/core/creation.hpp"   // zeros
#include "numpp/grids/grids.hpp"     // vander
#include "numpp/linalg/linalg.hpp"   // eigvals, lstsq
#include "numpp/umath/ufunc.hpp"     // real

#include <algorithm>
#include <cmath>
#include <vector>

namespace numpp {
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

// Pad the front (highest-degree side) of a coefficient vector to length n.
std::vector<double> pad_front(const std::vector<double>& v, size_t n) {
  std::vector<double> out(n, 0.0);
  std::copy(v.begin(), v.end(), out.begin() + (n - v.size()));
  return out;
}

// Discrete convolution of two coefficient/signal vectors (full).
std::vector<double> conv_full(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.empty() || b.empty()) return {};
  std::vector<double> out(a.size() + b.size() - 1, 0.0);
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < b.size(); ++j) out[i + j] += a[i] * b[j];
  return out;
}

std::vector<double> slice_mode(const std::vector<double>& full, size_t la, size_t lv, const std::string& mode) {
  if (mode == "full") return full;
  const size_t big = std::max(la, lv), small = std::min(la, lv);
  if (mode == "same") {
    const size_t start = (full.size() - big) / 2;
    return std::vector<double>(full.begin() + start, full.begin() + start + big);
  }
  if (mode == "valid") {
    const size_t len = big - small + 1;
    const size_t start = small - 1;
    return std::vector<double>(full.begin() + start, full.begin() + start + len);
  }
  throw value_error("convolve/correlate: mode must be full, same or valid");
}

}  // namespace

ndarray convolve(const ndarray& a, const ndarray& v, const std::string& mode) {
  std::vector<double> av = to_vec(a), vv = to_vec(v);
  return from_vec(slice_mode(conv_full(av, vv), av.size(), vv.size(), mode));
}

ndarray correlate(const ndarray& a, const ndarray& v, const std::string& mode) {
  std::vector<double> av = to_vec(a), vv = to_vec(v);
  std::vector<double> vr(vv.rbegin(), vv.rend());  // numpy: correlate(a,v) = convolve(a, reverse(v))
  return from_vec(slice_mode(conv_full(av, vr), av.size(), vr.size(), mode));
}

ndarray interp(const ndarray& x, const ndarray& xp, const ndarray& fp) {
  std::vector<double> xv = to_vec(x), xpv = to_vec(xp), fpv = to_vec(fp);
  const int64_t n = static_cast<int64_t>(xpv.size());
  ndarray out(x.shape(), kFloat64, Order::C);
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  for (size_t k = 0; k < xv.size(); ++k) {
    const double xi = xv[k];
    if (xi <= xpv.front()) { o[k] = fpv.front(); continue; }
    if (xi >= xpv.back()) { o[k] = fpv.back(); continue; }
    int64_t hi = 1;
    while (hi < n && xpv[hi] < xi) ++hi;
    const double t = (xi - xpv[hi - 1]) / (xpv[hi] - xpv[hi - 1]);
    o[k] = fpv[hi - 1] + t * (fpv[hi] - fpv[hi - 1]);
  }
  return out;
}

ndarray polyval(const ndarray& p, const ndarray& x) {
  std::vector<double> pv = to_vec(p);
  ndarray xf = x.astype(kFloat64).ascontiguousarray();
  ndarray out(xf.shape(), kFloat64, Order::C);
  const double* px = xf.size() ? xf.typed_data<double>() : nullptr;
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < xf.size(); ++i) {
    double r = 0.0;
    for (double c : pv) r = r * px[i] + c;
    o[i] = r;
  }
  return out;
}

ndarray polyadd(const ndarray& a, const ndarray& b) {
  std::vector<double> av = to_vec(a), bv = to_vec(b);
  const size_t n = std::max(av.size(), bv.size());
  av = pad_front(av, n); bv = pad_front(bv, n);
  std::vector<double> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = av[i] + bv[i];
  return from_vec(out);
}
ndarray polysub(const ndarray& a, const ndarray& b) {
  std::vector<double> av = to_vec(a), bv = to_vec(b);
  const size_t n = std::max(av.size(), bv.size());
  av = pad_front(av, n); bv = pad_front(bv, n);
  std::vector<double> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = av[i] - bv[i];
  return from_vec(out);
}
ndarray polymul(const ndarray& a, const ndarray& b) { return from_vec(conv_full(to_vec(a), to_vec(b))); }

std::pair<ndarray, ndarray> polydiv(const ndarray& a, const ndarray& b) {
  std::vector<double> u = to_vec(a), v = to_vec(b);
  // strip leading zeros of divisor
  size_t vs = 0; while (vs < v.size() && v[vs] == 0.0) ++vs;
  v.erase(v.begin(), v.begin() + vs);
  if (v.empty()) throw value_error("polydiv: divide by zero polynomial");
  if (u.size() < v.size()) return {from_vec({0.0}), from_vec(u)};
  std::vector<double> r = u;
  const size_t qn = u.size() - v.size() + 1;
  std::vector<double> q(qn, 0.0);
  for (size_t i = 0; i < qn; ++i) {
    const double coef = r[i] / v[0];
    q[i] = coef;
    for (size_t j = 0; j < v.size(); ++j) r[i + j] -= coef * v[j];
  }
  std::vector<double> rem(r.begin() + qn, r.end());  // degree < divisor
  if (rem.empty()) rem = {0.0};
  return {from_vec(q), from_vec(rem)};
}

ndarray polyder(const ndarray& p, int64_t m) {
  std::vector<double> c = to_vec(p);
  for (int64_t step = 0; step < m; ++step) {
    if (c.size() <= 1) { c = {0.0}; break; }
    const int64_t deg = static_cast<int64_t>(c.size()) - 1;
    std::vector<double> d(c.size() - 1);
    for (size_t i = 0; i < d.size(); ++i) d[i] = c[i] * static_cast<double>(deg - static_cast<int64_t>(i));
    c = d;
  }
  return from_vec(c);
}

ndarray polyint(const ndarray& p, int64_t m, double k) {
  std::vector<double> c = to_vec(p);
  for (int64_t step = 0; step < m; ++step) {
    const int64_t L = static_cast<int64_t>(c.size());  // current degree+1
    std::vector<double> ic(L + 1);
    for (int64_t i = 0; i < L; ++i) ic[i] = c[i] / static_cast<double>(L - i);  // x^(L-i)
    ic[L] = k;
    c = ic;
  }
  return from_vec(c);
}

ndarray poly(const ndarray& rts) {
  std::vector<double> r = to_vec(rts);
  std::vector<double> c = {1.0};
  for (double root : r) c = conv_full(c, {1.0, -root});
  return from_vec(c);
}

ndarray roots(const ndarray& p) {
  std::vector<double> c = to_vec(p);
  size_t s = 0; while (s < c.size() && c[s] == 0.0) ++s;  // strip leading zeros
  c.erase(c.begin(), c.begin() + s);
  // strip trailing zeros -> roots at the origin
  int64_t trailing = 0;
  while (c.size() > 1 && c.back() == 0.0) { c.pop_back(); ++trailing; }
  const int64_t N = static_cast<int64_t>(c.size()) - 1;
  std::vector<double> finite_roots;  // companion-matrix eigenvalues
  if (N >= 1) {
    ndarray comp = zeros({N, N}, kFloat64);
    for (int64_t j = 0; j < N; ++j) comp.set_item<double>({0, j}, -c[j + 1] / c[0]);
    for (int64_t i = 1; i < N; ++i) comp.set_item<double>({i, i - 1}, 1.0);
    ndarray ev = linalg::eigvals(comp);  // complex128
    ndarray evr = real(ev).astype(kFloat64).ascontiguousarray();
    const double* pe = evr.size() ? evr.typed_data<double>() : nullptr;
    for (int64_t i = 0; i < evr.size(); ++i) finite_roots.push_back(pe[i]);
  }
  for (int64_t i = 0; i < trailing; ++i) finite_roots.push_back(0.0);
  return from_vec(finite_roots);
}

ndarray polyfit(const ndarray& x, const ndarray& y, int64_t deg) {
  ndarray V = vander(x, deg + 1, false);  // decreasing powers, highest-first
  linalg::LstsqResult res = linalg::lstsq(V, y);
  return res.solution.astype(kFloat64);
}

}  // namespace numpp
