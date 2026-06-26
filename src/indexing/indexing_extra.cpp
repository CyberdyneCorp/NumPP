#include "numpp/indexing/indexing_extra.hpp"

#include "numpp/indexing/indexing.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/core/error.hpp"
#include "numpp/umath/ufunc.hpp"
#include <cstring>
#include <vector>


namespace numpp {

namespace {

// Odometer increment of a C-order multi-index over `shape`. False at wrap.
bool ic_next(std::vector<int64_t>& idx, const Shape& shape) {
  for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
    if (++idx[d] < shape[d]) return true;
    idx[d] = 0;
  }
  return false;
}

// Build a 1-D int64 array from a vector.
ndarray ic_from_i64(const std::vector<int64_t>& v) {
  ndarray out({static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i)
    out.set_item<int64_t>({static_cast<int64_t>(i)}, v[i]);
  return out;
}

}  // namespace

ndarray boolean_index(const ndarray& a, const ndarray& mask) {
  ndarray bm = mask.astype(kBool).broadcast_to(a.shape());
  const int64_t isz = a.dtype().itemsize();
  const Shape sh = a.shape();
  int64_t cnt = 0;
  if (a.size() > 0) {
    std::vector<int64_t> idx(sh.size(), 0);
    do {
      if (bm.item<bool>(idx)) ++cnt;
    } while (ic_next(idx, sh));
  }
  ndarray out({cnt}, a.dtype(), Order::C);
  if (cnt == 0) return out;
  char* o = out.bytes();
  int64_t w = 0;
  std::vector<int64_t> idx(sh.size(), 0);
  do {
    if (bm.item<bool>(idx)) {
      std::memcpy(o + (w++) * isz, a.element_ptr(idx), isz);
    }
  } while (ic_next(idx, sh));
  return out;
}

void boolean_assign(ndarray& a, const ndarray& mask, const ndarray& values) {
  if (a.size() == 0) return;
  ndarray bm = mask.astype(kBool).broadcast_to(a.shape());
  const int64_t isz = a.dtype().itemsize();
  ndarray vals = values.astype(a.dtype()).ascontiguousarray();
  vals = vals.reshape({vals.size()});
  const int64_t vn = vals.size();
  if (vn == 0) return;
  const char* v = vals.bytes();
  const Shape sh = a.shape();
  int64_t w = 0;
  std::vector<int64_t> idx(sh.size(), 0);
  do {
    if (bm.item<bool>(idx)) {
      std::memcpy(a.element_ptr(idx), v + (w % vn) * isz, isz);
      ++w;
    }
  } while (ic_next(idx, sh));
}

ndarray fancy_index(const ndarray& a, const ndarray& indices) {
  return take(a, indices);
}

void fancy_assign(ndarray& a, const ndarray& indices, const ndarray& values) {
  // a.flat[indices[k]] = values[k % nvalues] (C-order flat), matching numpy.put.
  put(a, indices, values);
}

void put_along_axis(ndarray& a, const ndarray& indices, const ndarray& values, int64_t axis) {
  const int64_t ax = normalize_axis(axis, a.ndim());
  const int64_t isz = a.dtype().itemsize();
  ndarray idx = indices.astype(kInt64);
  const Shape osh = idx.shape();
  if (idx.size() == 0) return;
  ndarray vals = values.astype(a.dtype()).broadcast_to(osh);
  std::vector<int64_t> oi(osh.size(), 0);
  do {
    int64_t k = idx.item<int64_t>(oi);
    if (k < 0) k += a.shape()[ax];
    std::vector<int64_t> ai = oi;
    ai[ax] = k;
    std::memcpy(a.element_ptr(ai), vals.element_ptr(oi), isz);
  } while (ic_next(oi, osh));
}

void place(ndarray& a, const ndarray& mask, const ndarray& vals) {
  if (a.size() == 0) return;
  ndarray m = mask.astype(kBool).ascontiguousarray();
  m = m.reshape({m.size()});
  const bool* c = m.size() ? m.typed_data<bool>() : nullptr;
  ndarray v = vals.astype(a.dtype()).ascontiguousarray();
  v = v.reshape({v.size()});
  const int64_t vn = v.size();
  if (vn == 0) return;
  const char* vp = v.bytes();
  const int64_t isz = a.dtype().itemsize();
  const int64_t mn = m.size();
  const Shape sh = a.shape();
  int64_t w = 0, f = 0;
  std::vector<int64_t> idx(sh.size(), 0);
  do {
    if (f < mn && c[f]) {
      std::memcpy(a.element_ptr(idx), vp + (w % vn) * isz, isz);
      ++w;
    }
    ++f;
  } while (ic_next(idx, sh));
}

std::vector<ndarray> ix_(const std::vector<ndarray>& seqs) {
  const int64_t n = static_cast<int64_t>(seqs.size());
  std::vector<ndarray> out;
  out.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    ndarray s = seqs[i].astype(kInt64).ascontiguousarray();
    Shape sh(static_cast<size_t>(n), 1);
    sh[i] = s.size();
    out.push_back(s.reshape(sh));
  }
  return out;
}

void fill_diagonal(ndarray& a, double val) {
  if (a.ndim() < 2) throw value_error("fill_diagonal: array must be at least 2-D");
  const int64_t isz = a.dtype().itemsize();
  ndarray cell = full({1}, val, a.dtype());
  const char* src = cell.bytes();
  if (a.ndim() == 2) {
    const int64_t len = std::min(a.shape()[0], a.shape()[1]);
    for (int64_t i = 0; i < len; ++i)
      std::memcpy(a.element_ptr({i, i}), src, isz);
    return;
  }
  int64_t len = a.shape()[0];
  for (int64_t d = 1; d < a.ndim(); ++d) len = std::min(len, a.shape()[d]);
  std::vector<int64_t> idx(static_cast<size_t>(a.ndim()));
  for (int64_t i = 0; i < len; ++i) {
    for (int64_t d = 0; d < a.ndim(); ++d) idx[d] = i;
    std::memcpy(a.element_ptr(idx), src, isz);
  }
}

std::vector<ndarray> diag_indices(int64_t n, int64_t ndim) {
  std::vector<ndarray> out;
  out.reserve(ndim);
  for (int64_t d = 0; d < ndim; ++d)
    out.push_back(arange(0.0, static_cast<double>(n), 1.0, kInt64));
  return out;
}

std::vector<ndarray> tril_indices(int64_t n, int64_t k, int64_t m) {
  if (m < 0) m = n;
  std::vector<int64_t> rows, cols;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < m; ++j)
      if (j <= i + k) { rows.push_back(i); cols.push_back(j); }
  return {ic_from_i64(rows), ic_from_i64(cols)};
}

std::vector<ndarray> triu_indices(int64_t n, int64_t k, int64_t m) {
  if (m < 0) m = n;
  std::vector<int64_t> rows, cols;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < m; ++j)
      if (j >= i + k) { rows.push_back(i); cols.push_back(j); }
  return {ic_from_i64(rows), ic_from_i64(cols)};
}

}  // namespace numpp
