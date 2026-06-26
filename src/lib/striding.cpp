#include "numpp/lib/stride_tricks.hpp"

#include "numpp/lib/leftovers.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/grids/grids.hpp"
#include "numpp/umath/ufunc.hpp"
#include "numpp/stats/stats.hpp"
#include "numpp/manip/manip.hpp"
#include "numpp/core/error.hpp"
#include <functional>
#include <vector>
#include <cstring>


namespace numpp {

namespace {

// Increment a row-major multi-index over `shape`. Returns false when wrapped past end.
inline bool incr_index(std::vector<int64_t>& idx, const Shape& shape) {
  for (int64_t ax = static_cast<int64_t>(shape.size()) - 1; ax >= 0; --ax) {
    if (++idx[ax] < shape[ax]) return true;
    idx[ax] = 0;
  }
  return false;
}

inline ndarray make_1d_i64(const std::vector<int64_t>& v) {
  ndarray out(Shape{static_cast<int64_t>(v.size())}, kInt64);
  for (size_t i = 0; i < v.size(); ++i) out.set_item<int64_t>({static_cast<int64_t>(i)}, v[i]);
  return out;
}

}  // namespace

ndarray sliding_window_view(const ndarray& a, const std::vector<int64_t>& window_shape) {
  const int64_t nd = a.ndim();
  const int64_t w = static_cast<int64_t>(window_shape.size());
  if (w > nd) throw value_error("window_shape cannot be larger than input ndim");
  const int64_t nlead = nd - w;
  const Shape& ash = a.shape();

  Shape out_shape;
  for (int64_t i = 0; i < nlead; ++i) out_shape.push_back(ash[i]);
  for (int64_t i = 0; i < w; ++i) {
    const int64_t len = ash[nlead + i] - window_shape[i] + 1;
    if (len < 0) throw value_error("window_shape is too large for input array");
    out_shape.push_back(len);
  }
  for (int64_t i = 0; i < w; ++i) out_shape.push_back(window_shape[i]);

  ndarray out(out_shape, a.dtype());
  const int64_t itemsize = a.itemsize();
  const int64_t out_nd = static_cast<int64_t>(out_shape.size());
  const int64_t total = out.size();
  if (total == 0) return out;

  char* obytes = out.bytes();
  std::vector<int64_t> oidx(static_cast<size_t>(out_nd), 0);
  std::vector<int64_t> sidx(static_cast<size_t>(nd), 0);
  int64_t flat = 0;
  while (true) {
    for (int64_t i = 0; i < nlead; ++i) sidx[i] = oidx[i];
    for (int64_t i = 0; i < w; ++i) sidx[nlead + i] = oidx[nlead + i] + oidx[nlead + w + i];
    const char* src = a.element_ptr(sidx);
    std::memcpy(obytes + flat * itemsize, src, static_cast<size_t>(itemsize));
    ++flat;
    if (!incr_index(oidx, out_shape)) break;
  }
  return out;
}

ndarray as_strided(const ndarray& a, const Shape& shape, const std::vector<int64_t>& strides_bytes) {
  ndarray out(shape, a.dtype());
  const int64_t itemsize = a.itemsize();
  const int64_t nd = static_cast<int64_t>(shape.size());
  const int64_t total = out.size();
  if (total == 0) return out;

  const char* abase = a.bytes();
  char* obytes = out.bytes();
  std::vector<int64_t> idx(static_cast<size_t>(nd), 0);
  int64_t flat = 0;
  while (true) {
    int64_t off = 0;
    for (int64_t d = 0; d < nd; ++d) off += idx[d] * strides_bytes[d];
    std::memcpy(obytes + flat * itemsize, abase + off, static_cast<size_t>(itemsize));
    ++flat;
    if (nd == 0) break;
    if (!incr_index(idx, shape)) break;
  }
  return out;
}

ndarray piecewise(const ndarray& x, const std::vector<ndarray>& condlist,
                  const std::vector<std::function<double(double)>>& funclist) {
  const Shape& xs = x.shape();
  ndarray xf = x.astype(kFloat64).ascontiguousarray();
  ndarray out = zeros(xs, kFloat64);
  const int64_t n = out.size();

  std::vector<ndarray> conds;
  conds.reserve(condlist.size());
  for (const auto& c : condlist) {
    conds.push_back(c.broadcast_to(xs).astype(kBool).ascontiguousarray());
  }

  const bool has_default = funclist.size() > condlist.size();
  if (n == 0) return out;

  const double* xd = xf.typed_data<double>();
  double* od = out.typed_data<double>();
  for (int64_t k = 0; k < n; ++k) {
    const double xv = xd[k];
    bool matched = false;
    const size_t lim = std::min(conds.size(), funclist.size());
    for (size_t i = 0; i < lim; ++i) {
      const unsigned char* cb = reinterpret_cast<const unsigned char*>(conds[i].bytes());
      if (cb[k]) {
        od[k] = funclist[i](xv);
        matched = true;
      }
    }
    if (!matched && has_default) od[k] = funclist.back()(xv);
  }
  return out;
}

ndarray apply_along_axis(const std::function<ndarray(const ndarray&)>& func, int64_t axis,
                         const ndarray& a) {
  const int64_t nd = a.ndim();
  const int64_t ax = normalize_axis(axis, nd);

  // Move the target axis to the last position.
  std::vector<int64_t> perm;
  perm.reserve(static_cast<size_t>(nd));
  for (int64_t d = 0; d < nd; ++d)
    if (d != ax) perm.push_back(d);
  perm.push_back(ax);
  ndarray at = a.transpose(perm).ascontiguousarray();

  Shape outer(at.shape().begin(), at.shape().end() - 1);
  int64_t nouter = 1;
  for (int64_t s : outer) nouter *= s;

  int64_t L = -1;
  std::vector<std::vector<double>> results;
  results.reserve(static_cast<size_t>(nouter));
  std::vector<int64_t> oidx(outer.size(), 0);
  for (int64_t t = 0; t < nouter; ++t) {
    std::vector<IndexItem> items;
    items.reserve(outer.size() + 1);
    for (size_t d = 0; d < outer.size(); ++d) items.push_back(IndexItem{oidx[d]});
    items.push_back(IndexItem{Slice{}});
    ndarray slice1d = at.index(items).ascontiguousarray();
    ndarray r = func(slice1d).astype(kFloat64).ravel().ascontiguousarray();
    const int64_t rl = r.size();
    if (L < 0) L = rl;
    std::vector<double> rv(static_cast<size_t>(rl));
    if (rl > 0) {
      const double* rp = r.typed_data<double>();
      for (int64_t i = 0; i < rl; ++i) rv[static_cast<size_t>(i)] = rp[i];
    }
    results.push_back(std::move(rv));
    incr_index(oidx, outer);
  }
  if (L < 0) L = 0;

  Shape rm_shape = outer;
  rm_shape.push_back(L);
  ndarray rm(rm_shape, kFloat64);
  if (rm.size() > 0) {
    double* rd = rm.typed_data<double>();
    int64_t flat = 0;
    for (int64_t t = 0; t < nouter; ++t) {
      const auto& rv = results[static_cast<size_t>(t)];
      for (int64_t i = 0; i < L; ++i)
        rd[flat++] = (i < static_cast<int64_t>(rv.size())) ? rv[static_cast<size_t>(i)] : 0.0;
    }
  }

  // Transpose the result axis (currently last) back into position `ax`.
  std::vector<int64_t> back(static_cast<size_t>(nd));
  for (int64_t d = 0; d < ax; ++d) back[d] = d;
  back[ax] = nd - 1;
  for (int64_t d = ax + 1; d < nd; ++d) back[d] = d - 1;
  ndarray result = rm.transpose(back).ascontiguousarray();

  // A scalar / length-1 per-slice result collapses (removes) that axis.
  if (L == 1) {
    const Shape& rs = result.shape();
    Shape ns;
    for (int64_t d = 0; d < static_cast<int64_t>(rs.size()); ++d)
      if (d != ax) ns.push_back(rs[d]);
    result = result.reshape(ns);
  }
  return result;
}

ndarray polyvander(const ndarray& x, int64_t deg) {
  return vander(x, deg + 1, /*increasing=*/true);
}

ndarray polycompanion(const ndarray& c) {
  ndarray cf = c.astype(kFloat64).ravel().ascontiguousarray();
  int64_t len = cf.size();
  const double* cp = cf.typed_data<double>();
  while (len > 1 && cp[len - 1] == 0.0) --len;
  if (len < 2) throw value_error("Series must have maximum degree of at least 1.");

  const int64_t n = len - 1;
  ndarray mat = zeros(Shape{n, n}, kFloat64);
  for (int64_t i = 1; i < n; ++i) mat.set_item<double>({i, i - 1}, 1.0);
  const double cn = cp[len - 1];
  for (int64_t i = 0; i < n; ++i) mat.set_item<double>({i, n - 1}, -cp[i] / cn);
  return mat;
}

std::vector<ndarray> mask_indices(int64_t n, const std::string& mask_func, int64_t k) {
  const bool tril = (mask_func == "tril");
  std::vector<int64_t> rows, cols;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const bool keep = tril ? (j <= i + k) : (j >= i + k);
      if (keep) {
        rows.push_back(i);
        cols.push_back(j);
      }
    }
  }
  return {make_1d_i64(rows), make_1d_i64(cols)};
}

}  // namespace numpp
