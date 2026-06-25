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

}  // namespace numpp
