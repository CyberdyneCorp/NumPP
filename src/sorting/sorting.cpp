#include "numpp/sorting/sorting.hpp"

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

}  // namespace numpp
