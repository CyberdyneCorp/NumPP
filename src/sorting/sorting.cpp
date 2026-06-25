#include "numpp/sorting/sorting.hpp"

#include "numpp/core/creation.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

namespace numpp {
namespace {

template <class T>
inline constexpr bool is_cplx = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

// numpy ordering: NaN sorts to the end; complex compares lexicographically.
template <class T>
bool less_nan(const T& a, const T& b) {
  if constexpr (is_cplx<T>) {
    if (a.real() != b.real()) return a.real() < b.real();
    return a.imag() < b.imag();
  } else if constexpr (std::is_floating_point_v<T>) {
    if (std::isnan(a)) return false;
    if (std::isnan(b)) return true;
    return a < b;
  } else {
    return a < b;
  }
}

bool stable(SortKind k) { return k == SortKind::Stable || k == SortKind::Mergesort; }

// half is sorted via a float32 proxy (it has no comparison operators).
DType sort_dtype(DType d) { return d == kFloat16 ? kFloat32 : d; }

// Move `ax` to the last position; returns contiguous copy with shape (outer, L).
ndarray to_axis_last(const ndarray& a, int64_t ax, int64_t& outer, int64_t& L, std::vector<int64_t>& perm) {
  const int64_t d = a.ndim();
  perm.clear();
  for (int64_t i = 0; i < d; ++i) if (i != ax) perm.push_back(i);
  perm.push_back(ax);
  ndarray moved = a.transpose(perm).ascontiguousarray();
  L = a.shape()[ax];
  outer = L > 0 ? a.size() / L : 0;
  return moved;
}
ndarray from_axis_last(const ndarray& moved, const std::vector<int64_t>& perm, int64_t d) {
  std::vector<int64_t> inv(d);
  for (int64_t i = 0; i < d; ++i) inv[perm[i]] = i;
  return moved.transpose(inv).ascontiguousarray();
}

}  // namespace

ndarray sort(const ndarray& a, std::optional<int64_t> axis, SortKind kind) {
  DType sd = sort_dtype(a.dtype());
  if (!axis) {  // flatten
    ndarray flat = a.astype(sd).ravel().copy();  // 1-D contiguous
    const int64_t n = flat.size();
    visit_dtype(sd.id(), [&](auto tag) {
      using T = typename decltype(tag)::type;
      if constexpr (std::is_same_v<T, half>) { (void)flat; }
      else { T* p = n ? flat.typed_data<T>() : nullptr; auto c = [](const T& x, const T& y) { return less_nan<T>(x, y); };
             if (stable(kind)) std::stable_sort(p, p + n, c); else std::sort(p, p + n, c); }
    });
    return flat.astype(a.dtype());
  }
  const int64_t ax = normalize_axis(*axis, a.ndim());
  int64_t outer = 0, L = 0; std::vector<int64_t> perm;
  ndarray moved = to_axis_last(a.astype(sd), ax, outer, L, perm);
  visit_dtype(sd.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half>) { (void)moved; }
    else {
      T* p = moved.size() ? moved.typed_data<T>() : nullptr;
      auto c = [](const T& x, const T& y) { return less_nan<T>(x, y); };
      for (int64_t r = 0; r < outer; ++r) { if (stable(kind)) std::stable_sort(p + r * L, p + r * L + L, c); else std::sort(p + r * L, p + r * L + L, c); }
    }
  });
  return from_axis_last(moved, perm, a.ndim()).astype(a.dtype());
}

ndarray argsort(const ndarray& a, std::optional<int64_t> axis, SortKind kind) {
  DType sd = sort_dtype(a.dtype());
  auto sort_row = [&](auto* vals, int64_t* idx, int64_t L) {
    using T = std::remove_pointer_t<decltype(vals)>;
    std::iota(idx, idx + L, int64_t{0});
    auto c = [&](int64_t i, int64_t j) { return less_nan<T>(vals[i], vals[j]); };
    if (stable(kind)) std::stable_sort(idx, idx + L, c); else std::sort(idx, idx + L, c);
  };
  if (!axis) {
    ndarray flat = a.astype(sd).ravel().copy();
    const int64_t n = flat.size();
    ndarray out(Shape{n}, kInt64, Order::C);
    int64_t* idx = n ? out.typed_data<int64_t>() : nullptr;
    visit_dtype(sd.id(), [&](auto tag) {
      using T = typename decltype(tag)::type;
      if constexpr (std::is_same_v<T, half>) { (void)idx; }
      else sort_row(flat.typed_data<T>(), idx, n);
    });
    return out;
  }
  const int64_t ax = normalize_axis(*axis, a.ndim());
  int64_t outer = 0, L = 0; std::vector<int64_t> perm;
  ndarray moved = to_axis_last(a.astype(sd), ax, outer, L, perm);
  ndarray out_moved(moved.shape(), kInt64, Order::C);
  int64_t* oi = out_moved.size() ? out_moved.typed_data<int64_t>() : nullptr;
  visit_dtype(sd.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half>) { (void)oi; }
    else { T* p = moved.size() ? moved.typed_data<T>() : nullptr; for (int64_t r = 0; r < outer; ++r) sort_row(p + r * L, oi + r * L, L); }
  });
  return from_axis_last(out_moved, perm, a.ndim());
}

ndarray searchsorted(const ndarray& a, const ndarray& v, const std::string& side) {
  DType ct = sort_dtype(result_type(a.dtype(), v.dtype()));
  ndarray ca = a.astype(ct).ravel().copy();
  ndarray cv = v.astype(ct).ascontiguousarray();
  ndarray out(v.shape(), kInt64, Order::C);
  const int64_t n = ca.size(), m = cv.size();
  const bool left = side != "right";
  visit_dtype(ct.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half>) { (void)out; }
    else {
      const T* base = n ? ca.typed_data<T>() : nullptr;
      const T* vv = m ? cv.typed_data<T>() : nullptr;
      int64_t* o = m ? out.typed_data<int64_t>() : nullptr;
      auto c = [](const T& x, const T& y) { return less_nan<T>(x, y); };
      for (int64_t i = 0; i < m; ++i) {
        const T* pos = left ? std::lower_bound(base, base + n, vv[i], c) : std::upper_bound(base, base + n, vv[i], c);
        o[i] = pos - base;
      }
    }
  });
  return out;
}

namespace {
ndarray select_axis(const ndarray& a, int64_t kth, std::optional<int64_t> axis, bool arg) {
  DType sd = sort_dtype(a.dtype());
  auto do_row = [&](auto* vals, int64_t* idx, int64_t L, int64_t k) {
    using T = std::remove_pointer_t<decltype(vals)>;
    if (k < 0) k += L;
    if (arg) {
      std::iota(idx, idx + L, int64_t{0});
      std::nth_element(idx, idx + k, idx + L, [&](int64_t i, int64_t j) { return less_nan<T>(vals[i], vals[j]); });
    } else {
      std::nth_element(vals, vals + k, vals + L, [](const T& x, const T& y) { return less_nan<T>(x, y); });
    }
  };
  if (!axis) {
    ndarray flat = a.astype(sd).ravel().copy();
    const int64_t n = flat.size();
    ndarray idx(Shape{n}, kInt64, Order::C);
    visit_dtype(sd.id(), [&](auto tag) {
      using T = typename decltype(tag)::type;
      if constexpr (std::is_same_v<T, half>) { (void)flat; }
      else do_row(flat.typed_data<T>(), n ? idx.typed_data<int64_t>() : nullptr, n, kth);
    });
    return arg ? idx : flat.astype(a.dtype());
  }
  const int64_t ax = normalize_axis(*axis, a.ndim());
  int64_t outer = 0, L = 0; std::vector<int64_t> perm;
  ndarray moved = to_axis_last(a.astype(sd), ax, outer, L, perm);
  ndarray idx(moved.shape(), kInt64, Order::C);
  visit_dtype(sd.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half>) { (void)moved; }
    else { T* p = moved.size() ? moved.typed_data<T>() : nullptr; int64_t* oi = idx.size() ? idx.typed_data<int64_t>() : nullptr;
           for (int64_t r = 0; r < outer; ++r) do_row(p + r * L, oi + r * L, L, kth); }
  });
  return arg ? from_axis_last(idx, perm, a.ndim()) : from_axis_last(moved, perm, a.ndim()).astype(a.dtype());
}

// argmin/argmax along an axis (or flattened). numpy returns the first NaN index.
template <class T>
int64_t arg_best(const T* p, int64_t L, bool want_max) {
  int64_t best = 0;
  for (int64_t i = 1; i < L; ++i) {
    bool replace;
    if constexpr (std::is_floating_point_v<T>) {
      bool bnan = std::isnan(p[best]), xnan = std::isnan(p[i]);
      replace = !bnan && (xnan || (want_max ? p[i] > p[best] : p[i] < p[best]));
    } else {
      replace = want_max ? p[i] > p[best] : p[i] < p[best];
    }
    if (replace) best = i;
  }
  return best;
}

ndarray arg_minmax(const ndarray& a, std::optional<int64_t> axis, bool want_max) {
  DType sd = sort_dtype(a.dtype());
  if (!axis) {
    ndarray flat = a.astype(sd).ravel().copy();
    int64_t r = 0;
    visit_dtype(sd.id(), [&](auto tag) {
      using T = typename decltype(tag)::type;
      if constexpr (std::is_same_v<T, half> || is_cplx<T>) { (void)flat; }
      else r = arg_best<T>(flat.typed_data<T>(), flat.size(), want_max);
    });
    ndarray out(Shape{}, kInt64, Order::C); out.set_item<int64_t>({}, r);
    return out;
  }
  const int64_t ax = normalize_axis(*axis, a.ndim());
  int64_t outer = 0, L = 0; std::vector<int64_t> perm;
  ndarray moved = to_axis_last(a.astype(sd), ax, outer, L, perm);
  Shape oshape(moved.shape().begin(), moved.shape().end() - 1);
  ndarray out(oshape, kInt64, Order::C);
  visit_dtype(sd.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half> || is_cplx<T>) { (void)out; }
    else { const T* p = moved.size() ? moved.typed_data<T>() : nullptr; int64_t* o = out.size() ? out.typed_data<int64_t>() : nullptr;
           for (int64_t r = 0; r < outer; ++r) o[r] = arg_best<T>(p + r * L, L, want_max); }
  });
  return out;
}

bool is_nonzero(const ndarray& c, int64_t i) {
  bool nz = false;
  visit_dtype(c.dtype().id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    const T v = c.typed_data<T>()[i];
    if constexpr (is_cplx<T>) nz = v.real() != 0 || v.imag() != 0;
    else if constexpr (std::is_same_v<T, half>) nz = static_cast<float>(v) != 0.0f;
    else nz = v != T{};
  });
  return nz;
}
}  // namespace

ndarray partition(const ndarray& a, int64_t kth, std::optional<int64_t> axis) { return select_axis(a, kth, axis, false); }
ndarray argpartition(const ndarray& a, int64_t kth, std::optional<int64_t> axis) { return select_axis(a, kth, axis, true); }
ndarray argmin(const ndarray& a, std::optional<int64_t> axis) { return arg_minmax(a, axis, false); }
ndarray argmax(const ndarray& a, std::optional<int64_t> axis) { return arg_minmax(a, axis, true); }

ndarray flatnonzero(const ndarray& a) {
  ndarray c = a.ascontiguousarray();
  std::vector<int64_t> idx;
  for (int64_t i = 0; i < c.size(); ++i) if (is_nonzero(c, i)) idx.push_back(i);
  ndarray out(Shape{(int64_t)idx.size()}, kInt64, Order::C);
  for (size_t i = 0; i < idx.size(); ++i) out.set_item<int64_t>({(int64_t)i}, idx[i]);
  return out;
}

ndarray count_nonzero(const ndarray& a, std::optional<int64_t> axis) {
  if (!axis) {
    ndarray c = a.ascontiguousarray();
    int64_t n = 0;
    for (int64_t i = 0; i < c.size(); ++i) if (is_nonzero(c, i)) ++n;
    ndarray out(Shape{}, kInt64, Order::C); out.set_item<int64_t>({}, n);
    return out;
  }
  const int64_t ax = normalize_axis(*axis, a.ndim());
  int64_t outer = 0, L = 0; std::vector<int64_t> perm;
  ndarray moved = to_axis_last(a, ax, outer, L, perm);
  Shape oshape(moved.shape().begin(), moved.shape().end() - 1);
  ndarray out(oshape, kInt64, Order::C);
  int64_t* o = out.size() ? out.typed_data<int64_t>() : nullptr;
  for (int64_t r = 0; r < outer; ++r) { int64_t cnt = 0; for (int64_t j = 0; j < L; ++j) if (is_nonzero(moved, r * L + j)) ++cnt; o[r] = cnt; }
  return out;
}

namespace {
template <class T> bool eq_nan(const T& x, const T& y) { return !less_nan<T>(x, y) && !less_nan<T>(y, x); }

// Concatenate two 1-D arrays of the same dtype.
ndarray cat1d(const ndarray& x, const ndarray& y) {
  ndarray out(Shape{x.size() + y.size()}, x.dtype(), Order::C);
  std::memcpy(out.bytes(), x.ascontiguousarray().bytes(), static_cast<size_t>(x.nbytes()));
  std::memcpy(out.bytes() + x.nbytes(), y.ascontiguousarray().bytes(), static_cast<size_t>(y.nbytes()));
  return out;
}
}  // namespace

UniqueResult unique_ex(const ndarray& a, bool ri, bool rinv, bool rc) {
  const DType sd = sort_dtype(a.dtype());
  ndarray flat = a.astype(sd).ravel().copy();
  const int64_t n = flat.size();
  ndarray perm = argsort(flat, std::nullopt, SortKind::Stable);
  UniqueResult res;
  visit_dtype(sd.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half>) { (void)flat; }
    else {
      const T* f = n ? flat.typed_data<T>() : nullptr;
      const int64_t* pm = n ? perm.typed_data<int64_t>() : nullptr;
      std::vector<int64_t> upos;
      for (int64_t i = 0; i < n; ++i) if (i == 0 || !eq_nan<T>(f[pm[i]], f[pm[i - 1]])) upos.push_back(i);
      const int64_t k = static_cast<int64_t>(upos.size());
      ndarray vals(Shape{k}, sd, Order::C);
      T* vp = k ? vals.typed_data<T>() : nullptr;
      for (int64_t j = 0; j < k; ++j) vp[j] = f[pm[upos[j]]];
      res.values = vals.astype(a.dtype());
      if (ri) { res.index = ndarray(Shape{k}, kInt64, Order::C); for (int64_t j = 0; j < k; ++j) res.index.set_item<int64_t>({j}, pm[upos[j]]); }
      if (rc) { res.counts = ndarray(Shape{k}, kInt64, Order::C); for (int64_t j = 0; j < k; ++j) res.counts.set_item<int64_t>({j}, (j + 1 < k ? upos[j + 1] : n) - upos[j]); }
      if (rinv) {
        res.inverse = ndarray(Shape{n}, kInt64, Order::C);
        int64_t u = -1; size_t up = 0;
        for (int64_t i = 0; i < n; ++i) { if (up < upos.size() && upos[up] == i) { ++u; ++up; } res.inverse.set_item<int64_t>({pm[i]}, u); }
      }
    }
  });
  return res;
}
ndarray unique(const ndarray& a) { return unique_ex(a, false, false, false).values; }

ndarray in1d(const ndarray& a, const ndarray& b) {
  const DType ct = sort_dtype(result_type(a.dtype(), b.dtype()));
  ndarray aa = a.astype(ct).ravel().copy();
  ndarray bb = b.astype(ct).ravel().copy();
  ndarray out(Shape{aa.size()}, kBool, Order::C);
  visit_dtype(ct.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half>) { (void)out; }
    else {
      T* bp = bb.size() ? bb.typed_data<T>() : nullptr;
      auto c = [](const T& x, const T& y) { return less_nan<T>(x, y); };
      std::sort(bp, bp + bb.size(), c);
      const T* ap = aa.size() ? aa.typed_data<T>() : nullptr;
      bool* o = aa.size() ? out.typed_data<bool>() : nullptr;
      for (int64_t i = 0; i < aa.size(); ++i) o[i] = std::binary_search(bp, bp + bb.size(), ap[i], c);
    }
  });
  return out;
}
ndarray isin(const ndarray& a, const ndarray& b) { return in1d(a, b).reshape(a.shape()); }

ndarray intersect1d(const ndarray& a, const ndarray& b) {
  const DType ct = sort_dtype(result_type(a.dtype(), b.dtype()));
  ndarray ua = unique(a).astype(ct), ub = unique(b).astype(ct);
  ndarray out;
  visit_dtype(ct.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half>) {}
    else {
      const T* pa = ua.size() ? ua.typed_data<T>() : nullptr;
      const T* pb = ub.size() ? ub.typed_data<T>() : nullptr;
      std::vector<T> common;
      int64_t i = 0, j = 0;
      while (i < ua.size() && j < ub.size()) {
        if (less_nan<T>(pa[i], pb[j])) ++i;
        else if (less_nan<T>(pb[j], pa[i])) ++j;
        else { common.push_back(pa[i]); ++i; ++j; }
      }
      out = ndarray(Shape{(int64_t)common.size()}, ct, Order::C);
      T* op = common.empty() ? nullptr : out.typed_data<T>();
      for (size_t t = 0; t < common.size(); ++t) op[t] = common[t];
    }
  });
  return out.astype(result_type(a.dtype(), b.dtype()));
}
ndarray union1d(const ndarray& a, const ndarray& b) {
  const DType ct = sort_dtype(result_type(a.dtype(), b.dtype()));
  return unique(cat1d(a.astype(ct).ravel().copy(), b.astype(ct).ravel().copy())).astype(result_type(a.dtype(), b.dtype()));
}
ndarray setdiff1d(const ndarray& a, const ndarray& b) {
  const DType ct = sort_dtype(result_type(a.dtype(), b.dtype()));
  ndarray ua = unique(a).astype(ct), ub = unique(b).astype(ct);
  ndarray out;
  visit_dtype(ct.id(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, half>) {}
    else {
      const T* pa = ua.size() ? ua.typed_data<T>() : nullptr;
      const T* pb = ub.size() ? ub.typed_data<T>() : nullptr;
      auto c = [](const T& x, const T& y) { return less_nan<T>(x, y); };
      std::vector<T> diff;
      for (int64_t i = 0; i < ua.size(); ++i) if (!std::binary_search(pb, pb + ub.size(), pa[i], c)) diff.push_back(pa[i]);
      out = ndarray(Shape{(int64_t)diff.size()}, ct, Order::C);
      T* op = diff.empty() ? nullptr : out.typed_data<T>();
      for (size_t t = 0; t < diff.size(); ++t) op[t] = diff[t];
    }
  });
  return out.astype(result_type(a.dtype(), b.dtype()));
}

ndarray bincount(const ndarray& x, const ndarray* weights, int64_t minlength) {
  ndarray xi = x.astype(kInt64).ravel().copy();
  const int64_t n = xi.size();
  const int64_t* p = n ? xi.typed_data<int64_t>() : nullptr;
  int64_t mx = -1;
  for (int64_t i = 0; i < n; ++i) { if (p[i] < 0) throw value_error("bincount: negative value"); mx = std::max(mx, p[i]); }
  const int64_t len = std::max(mx + 1, minlength);
  if (weights) {
    ndarray w = weights->astype(kFloat64).ravel().copy();
    ndarray out = zeros({len}, kFloat64);
    double* o = len ? out.typed_data<double>() : nullptr;
    const double* wp = n ? w.typed_data<double>() : nullptr;
    for (int64_t i = 0; i < n; ++i) o[p[i]] += wp[i];
    return out;
  }
  ndarray out = zeros({len}, kInt64);
  int64_t* o = len ? out.typed_data<int64_t>() : nullptr;
  for (int64_t i = 0; i < n; ++i) ++o[p[i]];
  return out;
}

Histogram histogram(const ndarray& a, int64_t bins, std::optional<std::pair<double, double>> range) {
  ndarray flat = a.astype(kFloat64).ravel().copy();
  const int64_t n = flat.size();
  const double* p = n ? flat.typed_data<double>() : nullptr;
  double lo, hi;
  if (range) { lo = range->first; hi = range->second; }
  else if (n == 0) { lo = 0; hi = 1; }
  else { lo = hi = p[0]; for (int64_t i = 1; i < n; ++i) { lo = std::min(lo, p[i]); hi = std::max(hi, p[i]); } }
  if (lo == hi) { lo -= 0.5; hi += 0.5; }
  ndarray edges = linspace(lo, hi, bins + 1);
  ndarray counts = zeros({bins}, kInt64);
  int64_t* c = bins ? counts.typed_data<int64_t>() : nullptr;
  const double norm = static_cast<double>(bins) / (hi - lo);
  for (int64_t i = 0; i < n; ++i) {
    double v = p[i];
    if (v < lo || v > hi) continue;
    int64_t b = v == hi ? bins - 1 : static_cast<int64_t>((v - lo) * norm);
    if (b >= 0 && b < bins) ++c[b];
  }
  return {counts, edges};
}
ndarray histogram_bin_edges(const ndarray& a, int64_t bins, std::optional<std::pair<double, double>> range) {
  return histogram(a, bins, range).bin_edges;
}

}  // namespace numpp
