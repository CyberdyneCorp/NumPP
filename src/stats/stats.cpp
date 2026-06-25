#include "numpp/stats/stats.hpp"

#include "numpp/backend/backend.hpp"     // matmul
#include "numpp/core/creation.hpp"       // zeros/full/arange
#include "numpp/linalg/linalg.hpp"       // outer
#include "numpp/core/shape.hpp"          // normalize_axis
#include "numpp/sorting/sorting.hpp"     // sort/argmin/argmax/searchsorted
#include "numpp/umath/ufunc.hpp"         // mean/sum/amax/amin/sqrt/where/isnan/clip/...

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace numpp {
namespace {

// Move `ax` to the last position; returns a contiguous copy of shape (outer, L).
ndarray to_last(const ndarray& a, int64_t ax, int64_t& outer, int64_t& L, std::vector<int64_t>& perm) {
  const int64_t d = a.ndim();
  perm.clear();
  for (int64_t i = 0; i < d; ++i) if (i != ax) perm.push_back(i);
  perm.push_back(ax);
  ndarray moved = a.transpose(perm).ascontiguousarray();
  L = a.shape()[ax];
  outer = L ? a.size() / L : 0;
  return moved;
}
ndarray from_last(const ndarray& moved, const std::vector<int64_t>& perm, int64_t d) {
  std::vector<int64_t> inv(d);
  for (int64_t i = 0; i < d; ++i) inv[perm[i]] = i;
  return moved.transpose(inv).ascontiguousarray();
}

void accumulate(ndarray& m, int64_t outer, int64_t L, bool prod) {
  visit_dtype(m.dtype().id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bool>) { (void)m; }
    else {
      T* p = m.size() ? m.typed_data<T>() : nullptr;
      for (int64_t r = 0; r < outer; ++r) {
        T acc = prod ? T(1) : T(0);
        for (int64_t j = 0; j < L; ++j) { acc = prod ? static_cast<T>(acc * p[r * L + j]) : static_cast<T>(acc + p[r * L + j]); p[r * L + j] = acc; }
      }
    }
  });
}

ndarray cumulative(const ndarray& a, std::optional<int64_t> axis, bool prod) {
  const DType base = a.dtype() == kBool ? kInt64 : a.dtype();
  const DType cdt = base == kFloat16 ? kFloat32 : base;  // accumulate off half/bool
  if (!axis) {
    ndarray f = a.astype(cdt).ravel().copy();
    accumulate(f, 1, f.size(), prod);
    return f.astype(base);
  }
  const int64_t ax = normalize_axis(*axis, a.ndim());
  int64_t outer = 0, L = 0; std::vector<int64_t> perm;
  ndarray moved = to_last(a.astype(cdt), ax, outer, L, perm);
  accumulate(moved, outer, L, prod);
  return from_last(moved, perm, a.ndim()).astype(base);
}

// Reduce each 1-D slice (along axis, or the whole flattened array) to a double.
ndarray reduce_axis_double(const ndarray& a, std::optional<int64_t> axis, const std::function<double(std::vector<double>&)>& f) {
  if (!axis) {
    ndarray fl = a.astype(kFloat64).ravel().copy();
    std::vector<double> v(fl.size() ? fl.typed_data<double>() : nullptr, fl.size() ? fl.typed_data<double>() + fl.size() : nullptr);
    ndarray out(Shape{}, kFloat64, Order::C); out.set_item<double>({}, f(v));
    return out;
  }
  const int64_t ax = normalize_axis(*axis, a.ndim());
  int64_t outer = 0, L = 0; std::vector<int64_t> perm;
  ndarray moved = to_last(a.astype(kFloat64), ax, outer, L, perm);
  Shape osh(moved.shape().begin(), moved.shape().end() - 1);
  ndarray out(osh, kFloat64, Order::C);
  const double* src = moved.size() ? moved.typed_data<double>() : nullptr;
  double* o = out.size() ? out.typed_data<double>() : nullptr;
  for (int64_t r = 0; r < outer; ++r) { std::vector<double> v(src + r * L, src + r * L + L); o[r] = f(v); }
  return out;
}

double pct_linear(std::vector<double> v, double q) {  // q in [0,100], sorts in place
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const double pos = q / 100.0 * static_cast<double>(v.size() - 1);
  const int64_t lo = static_cast<int64_t>(std::floor(pos)), hi = static_cast<int64_t>(std::ceil(pos));
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}
std::vector<double> drop_nan(const std::vector<double>& v) {
  std::vector<double> w; for (double x : v) if (!std::isnan(x)) w.push_back(x); return w;
}

}  // namespace

ndarray cumsum(const ndarray& a, std::optional<int64_t> axis) { return cumulative(a, axis, false); }
ndarray cumprod(const ndarray& a, std::optional<int64_t> axis) { return cumulative(a, axis, true); }
ndarray nancumsum(const ndarray& a, std::optional<int64_t> axis) { return cumulative(where(isnan(a), zeros_like(a), a), axis, false); }
ndarray nancumprod(const ndarray& a, std::optional<int64_t> axis) { return cumulative(where(isnan(a), full_like(a, 1.0), a), axis, true); }

ndarray diff(const ndarray& a, int64_t n, int64_t axis) {
  const int64_t ax = normalize_axis(axis, a.ndim());
  ndarray r = a;
  for (int64_t k = 0; k < n; ++k) {
    const int64_t len = r.shape()[ax];
    std::vector<IndexItem> hi, lo;
    for (int64_t d = 0; d < r.ndim(); ++d) {
      if (d == ax) { hi.push_back(IndexItem{Slice{1, len, 1}}); lo.push_back(IndexItem{Slice{0, len - 1, 1}}); }
      else { hi.push_back(IndexItem{Slice{}}); lo.push_back(IndexItem{Slice{}}); }
    }
    r = subtract(r.index(hi), r.index(lo));
  }
  return r;
}
ndarray ediff1d(const ndarray& a) { return diff(a.ravel(), 1, 0); }

ndarray gradient(const ndarray& a) {
  ndarray x = a.astype(kFloat64).ravel().copy();
  const int64_t n = x.size();
  ndarray g(Shape{n}, kFloat64, Order::C);
  const double* p = n ? x.typed_data<double>() : nullptr;
  double* o = n ? g.typed_data<double>() : nullptr;
  if (n == 1) { o[0] = 0.0; return g; }
  o[0] = p[1] - p[0];
  o[n - 1] = p[n - 1] - p[n - 2];
  for (int64_t i = 1; i < n - 1; ++i) o[i] = (p[i + 1] - p[i - 1]) / 2.0;
  return g;
}
ndarray ptp(const ndarray& a, std::optional<int64_t> axis) { return subtract(amax(a, axis, false), amin(a, axis, false)); }

ndarray percentile(const ndarray& a, double q, std::optional<int64_t> axis) {
  return reduce_axis_double(a, axis, [q](std::vector<double>& v) { return pct_linear(v, q); });
}
ndarray quantile(const ndarray& a, double q, std::optional<int64_t> axis) { return percentile(a, q * 100.0, axis); }
ndarray median(const ndarray& a, std::optional<int64_t> axis) { return percentile(a, 50.0, axis); }
ndarray nanpercentile(const ndarray& a, double q, std::optional<int64_t> axis) {
  return reduce_axis_double(a, axis, [q](std::vector<double>& v) { auto w = drop_nan(v); return pct_linear(w, q); });
}
ndarray nanmedian(const ndarray& a, std::optional<int64_t> axis) { return nanpercentile(a, 50.0, axis); }

ndarray average(const ndarray& a, std::optional<int64_t> axis, const ndarray* weights) {
  if (!weights) return mean(a, axis);
  ndarray af = a.astype(kFloat64);
  ndarray wb;
  if (weights->shape() == a.shape()) {
    wb = weights->astype(kFloat64);
  } else if (weights->ndim() == 1 && axis) {
    const int64_t ax = normalize_axis(*axis, a.ndim());
    Shape ws(a.ndim(), 1); ws[ax] = weights->size();
    wb = weights->astype(kFloat64).reshape(ws).broadcast_to(a.shape());
  } else if (weights->ndim() == 1 && !axis && weights->size() == a.size()) {
    ndarray wf = weights->astype(kFloat64);
    return divide(sum(multiply(af.ravel(), wf)), sum(wf));
  } else {
    throw value_error("average: weights shape mismatch");
  }
  return divide(sum(multiply(af, wb), axis), sum(wb, axis));
}

ndarray cov(const ndarray& m, bool rowvar, int64_t ddof) {
  const bool was_vec = m.ndim() == 1;
  ndarray X = m.astype(kFloat64);
  if (was_vec) X = X.reshape({1, X.size()});
  if (!rowvar) X = X.transpose().copy();
  const int64_t nobs = X.shape()[1];
  ndarray Xc = subtract(X, mean(X, 1, true));
  ndarray c = divide(matmul(Xc, Xc.transpose()), full(Shape{}, static_cast<double>(nobs - ddof), kFloat64));
  return was_vec ? c.reshape({}) : c;
}
ndarray corrcoef(const ndarray& m, bool rowvar) {
  ndarray c = cov(m, rowvar, 1);
  if (c.ndim() < 2) { ndarray one(Shape{}, kFloat64, Order::C); one.set_item<double>({}, 1.0); return one; }
  const int64_t n = c.shape()[0];
  ndarray d(Shape{n}, kFloat64, Order::C);
  for (int64_t i = 0; i < n; ++i) d.set_item<double>({i}, std::sqrt(c.item<double>({i, i})));
  ndarray r = divide(c, outer(d, d));
  return clip(r, full(r.shape(), -1.0, kFloat64), full(r.shape(), 1.0, kFloat64));
}

ndarray digitize(const ndarray& x, const ndarray& bins, bool right) {
  return searchsorted(bins, x, right ? "left" : "right");
}

ndarray nanargmin(const ndarray& a, std::optional<int64_t> axis) {
  const double inf = std::numeric_limits<double>::infinity();
  return argmin(where(isnan(a), full(a.shape(), inf, kFloat64), a.astype(kFloat64)), axis);
}
ndarray nanargmax(const ndarray& a, std::optional<int64_t> axis) {
  const double inf = std::numeric_limits<double>::infinity();
  return argmax(where(isnan(a), full(a.shape(), -inf, kFloat64), a.astype(kFloat64)), axis);
}

}  // namespace numpp
