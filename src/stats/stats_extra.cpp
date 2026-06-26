#include "numpp/stats/stats_extra.hpp"

#include "numpp/core/creation.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/umath/ufunc.hpp"
#include "numpp/sorting/sorting.hpp"
#include "numpp/backend/backend.hpp"
#include "numpp/core/error.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>


namespace numpp {

namespace {

// Build (lo, hi) range for 1-D values, matching numpy's degenerate handling.
std::pair<double, double> value_range(const double* p, int64_t n) {
  if (n == 0) return {0.0, 1.0};
  double lo = p[0], hi = p[0];
  for (int64_t i = 1; i < n; ++i) {
    if (p[i] < lo) lo = p[i];
    if (p[i] > hi) hi = p[i];
  }
  if (lo == hi) { lo -= 0.5; hi += 0.5; }
  return {lo, hi};
}

// Bin index for value v given (bins+1) uniform edges; numpy: last bin closed.
int64_t find_bin(double v, const std::vector<double>& edges, int64_t bins) {
  if (v < edges.front() || v > edges.back()) return -1;  // outside range: dropped
  auto it = std::upper_bound(edges.begin(), edges.end(), v);  // searchsorted 'right'
  int64_t b = (it - edges.begin()) - 1;
  if (b == bins) b = bins - 1;  // v == last (closed) edge
  if (b < 0) b = 0;
  return b;
}

std::vector<double> to_vec_f64(const ndarray& a) {
  ndarray f = a.astype(kFloat64).ravel().copy();
  const int64_t n = f.size();
  const double* p = n ? f.typed_data<double>() : nullptr;
  return std::vector<double>(p, p + n);
}

std::vector<double> edges_of(double lo, double hi, int64_t bins) {
  ndarray e = linspace(lo, hi, bins + 1, true, kFloat64);
  const double* p = e.typed_data<double>();
  return std::vector<double>(p, p + bins + 1);
}

ndarray edges_to_array(const std::vector<double>& e) {
  ndarray out(Shape{static_cast<int64_t>(e.size())}, kFloat64, Order::C);
  double* o = out.typed_data<double>();
  for (size_t i = 0; i < e.size(); ++i) o[i] = e[i];
  return out;
}

double quantile_1d(std::vector<double> v, double q) {  // q in [0,1]
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const double pos = q * static_cast<double>(v.size() - 1);
  const int64_t lo = static_cast<int64_t>(std::floor(pos));
  const int64_t hi = static_cast<int64_t>(std::ceil(pos));
  return v[lo] + (v[hi] - v[lo]) * (pos - static_cast<double>(lo));
}

std::vector<double> drop_nans(const std::vector<double>& v) {
  std::vector<double> w;
  for (double x : v) if (!std::isnan(x)) w.push_back(x);
  return w;
}

}  // namespace

Histogram2DResult histogram2d(const ndarray& x, const ndarray& y, int64_t bins) {
  if (bins < 1) throw value_error("histogram2d: bins must be >= 1");
  std::vector<double> xs = to_vec_f64(x);
  std::vector<double> ys = to_vec_f64(y);
  if (xs.size() != ys.size()) throw value_error("histogram2d: x and y must have the same length");
  const int64_t n = static_cast<int64_t>(xs.size());

  auto xr = value_range(xs.data(), n);
  auto yr = value_range(ys.data(), n);
  std::vector<double> xe = edges_of(xr.first, xr.second, bins);
  std::vector<double> ye = edges_of(yr.first, yr.second, bins);

  ndarray H = zeros({bins, bins}, kFloat64);
  double* h = H.typed_data<double>();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t bx = find_bin(xs[i], xe, bins);
    const int64_t by = find_bin(ys[i], ye, bins);
    if (bx >= 0 && by >= 0) h[bx * bins + by] += 1.0;
  }
  return Histogram2DResult{H, edges_to_array(xe), edges_to_array(ye)};
}

HistogramDDResult histogramdd(const ndarray& sample, int64_t bins) {
  if (bins < 1) throw value_error("histogramdd: bins must be >= 1");
  if (sample.ndim() != 2) throw value_error("histogramdd: sample must be (N, D)");
  const int64_t N = sample.shape()[0];
  const int64_t D = sample.shape()[1];

  ndarray sc = sample.astype(kFloat64).ascontiguousarray();
  const double* sp = sc.size() ? sc.typed_data<double>() : nullptr;

  // Per-dimension edges.
  std::vector<std::vector<double>> edges(D);
  for (int64_t d = 0; d < D; ++d) {
    std::vector<double> col(N);
    for (int64_t i = 0; i < N; ++i) col[i] = sp[i * D + d];
    auto r = value_range(col.data(), N);
    edges[d] = edges_of(r.first, r.second, bins);
  }

  Shape hshape(static_cast<size_t>(D), bins);
  ndarray H = zeros(hshape, kFloat64);
  double* h = H.size() ? H.typed_data<double>() : nullptr;

  // Row-major strides over the D-dimensional count array.
  std::vector<int64_t> strides(D, 1);
  for (int64_t d = D - 2; d >= 0; --d) strides[d] = strides[d + 1] * bins;

  std::vector<int64_t> idx(D);
  for (int64_t i = 0; i < N; ++i) {
    bool inside = true;
    for (int64_t d = 0; d < D && inside; ++d) {
      const int64_t b = find_bin(sp[i * D + d], edges[d], bins);
      if (b < 0) inside = false; else idx[d] = b;
    }
    if (!inside) continue;
    int64_t flat = 0;
    for (int64_t d = 0; d < D; ++d) flat += idx[d] * strides[d];
    h[flat] += 1.0;
  }

  std::vector<ndarray> ea;
  ea.reserve(D);
  for (int64_t d = 0; d < D; ++d) ea.push_back(edges_to_array(edges[d]));
  return HistogramDDResult{H, std::move(ea)};
}

ndarray nanquantile(const ndarray& a, double q, std::optional<int64_t> axis) {
  if (!axis) {
    std::vector<double> v = drop_nans(to_vec_f64(a));
    ndarray out(Shape{}, kFloat64, Order::C);
    out.set_item<double>({}, quantile_1d(std::move(v), q));
    return out;
  }
  const int64_t ax = normalize_axis(*axis, a.ndim());
  const int64_t d = a.ndim();
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < d; ++i) if (i != ax) perm.push_back(i);
  perm.push_back(ax);
  ndarray moved = a.astype(kFloat64).transpose(perm).ascontiguousarray();
  const int64_t L = a.shape()[ax];
  const int64_t outer = L ? a.size() / L : 0;

  Shape osh(moved.shape().begin(), moved.shape().end() - 1);
  ndarray out(osh, kFloat64, Order::C);
  const double* src = moved.size() ? moved.typed_data<double>() : nullptr;
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t r = 0; r < outer; ++r) {
    std::vector<double> v(src + r * L, src + r * L + L);
    o[r] = quantile_1d(drop_nans(v), q);
  }
  return out;
}

ndarray cov_weighted(const ndarray& m, const ndarray& aweights, bool rowvar, int64_t ddof) {
  const bool was_vec = m.ndim() == 1;
  ndarray X = m.astype(kFloat64);
  if (was_vec) X = X.reshape({1, X.size()});
  if (!rowvar) X = X.transpose().copy();
  const int64_t nvar = X.shape()[0];
  const int64_t nobs = X.shape()[1];

  ndarray w = aweights.astype(kFloat64).ravel().copy();
  if (w.size() != nobs) throw value_error("cov_weighted: aweights length must match observations");
  ndarray wrow = w.reshape({1, nobs});  // broadcasts across rows

  const double v1 = sum(w).item<double>({});
  const double v2 = sum(multiply(w, w)).item<double>({});
  const double fact = v1 - static_cast<double>(ddof) * v2 / v1;

  ndarray wmean = divide(sum(multiply(X, wrow), 1, true), full(Shape{}, v1, kFloat64));
  ndarray Xc = subtract(X, wmean);
  ndarray Xw = multiply(Xc, wrow);  // weight each observation column
  ndarray c = divide(matmul(Xw, Xc.transpose()), full(Shape{}, fact, kFloat64));
  (void)nvar;
  return was_vec ? c.reshape({}) : c;
}

}  // namespace numpp
