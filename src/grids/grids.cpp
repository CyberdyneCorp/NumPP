#include "numpp/grids/grids.hpp"

#include "numpp/core/creation.hpp"   // zeros/arange/linspace
#include "numpp/umath/ufunc.hpp"     // power/exp/log/sign/multiply/where/copyto

#include <cmath>
#include <cstdint>

namespace numpp {
namespace {

// Row-major scalar index from a multi-index over `shape`.
int64_t ravel_index(const std::vector<int64_t>& idx, const Shape& shape) {
  int64_t flat = 0;
  for (size_t d = 0; d < shape.size(); ++d) flat = flat * shape[d] + idx[d];
  return flat;
}

}  // namespace

std::vector<ndarray> meshgrid(const std::vector<ndarray>& xi, bool indexing_xy) {
  const int64_t n = static_cast<int64_t>(xi.size());
  Shape full(n);
  for (int64_t i = 0; i < n; ++i) full[i] = xi[i].size();  // 'ij' shape
  std::vector<ndarray> out;
  out.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    Shape rshape(n, 1);
    rshape[i] = xi[i].size();
    ndarray g = xi[i].reshape(rshape).broadcast_to(full).copy();
    if (indexing_xy && n >= 2) {
      std::vector<int64_t> perm(n);
      for (int64_t d = 0; d < n; ++d) perm[d] = d;
      perm[0] = 1; perm[1] = 0;
      g = g.transpose(perm).ascontiguousarray();
    }
    out.push_back(g);
  }
  return out;
}

ndarray indices(const Shape& shape) {
  const int64_t nd = static_cast<int64_t>(shape.size());
  Shape oshape;
  oshape.push_back(nd);
  for (int64_t s : shape) oshape.push_back(s);
  ndarray out = zeros(oshape, kInt64);
  for (int64_t d = 0; d < nd; ++d) {
    Shape rshape(nd, 1);
    rshape[d] = shape[d];
    ndarray g = arange(0.0, static_cast<double>(shape[d]), 1.0, kInt64).reshape(rshape).broadcast_to(shape);
    ndarray dst = out.index({IndexItem{d}});  // view of shape == shape
    copyto(dst, g);
  }
  return out;
}

ndarray diag(const ndarray& v, int64_t k) {
  if (v.ndim() == 1) {
    const int64_t len = v.size();
    const int64_t N = len + std::llabs(k);
    ndarray out = zeros({N, N}, v.dtype());
    for (int64_t i = 0; i < len; ++i) {
      const int64_t r = k >= 0 ? i : i - k;
      const int64_t c = k >= 0 ? i + k : i;
      out.set_item<double>({r, c}, v.item<double>({i}));
    }
    return out;
  }
  if (v.ndim() == 2) {
    const int64_t rows = v.shape()[0], cols = v.shape()[1];
    int64_t len = 0;
    if (k >= 0) len = std::min(rows, cols - k);
    else len = std::min(rows + k, cols);
    if (len < 0) len = 0;
    ndarray out = zeros({len}, v.dtype());
    for (int64_t i = 0; i < len; ++i) {
      const int64_t r = k >= 0 ? i : i - k;
      const int64_t c = k >= 0 ? i + k : i;
      out.set_item<double>({i}, v.item<double>({r, c}));
    }
    return out;
  }
  throw value_error("diag: input must be 1-D or 2-D");
}

ndarray diagflat(const ndarray& v, int64_t k) { return diag(v.ravel(), k); }

ndarray tri(int64_t n, int64_t m, int64_t k, DType dtype) {
  if (m < 0) m = n;
  ndarray out = zeros({n, m}, dtype);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < m; ++j)
      if (j <= i + k) out.set_item<double>({i, j}, 1.0);
  return out;
}

namespace {
// Apply a lower/upper triangular mask over the last two axes of `m`.
ndarray triangular(const ndarray& m, int64_t k, bool lower) {
  if (m.ndim() < 2) throw value_error("tril/triu: input must be at least 2-D");
  const int64_t R = m.shape()[m.ndim() - 2], C = m.shape()[m.ndim() - 1];
  ndarray mask2d = zeros({R, C}, kBool);
  for (int64_t i = 0; i < R; ++i)
    for (int64_t j = 0; j < C; ++j) {
      const bool keep = lower ? (j <= i + k) : (j >= i + k);
      if (keep) mask2d.set_item<bool>({i, j}, true);
    }
  ndarray mask = mask2d.broadcast_to(m.shape());
  return where(mask, m, zeros(m.shape(), m.dtype()));
}
}  // namespace

ndarray tril(const ndarray& m, int64_t k) { return triangular(m, k, true); }
ndarray triu(const ndarray& m, int64_t k) { return triangular(m, k, false); }

ndarray vander(const ndarray& x, int64_t n, bool increasing) {
  ndarray xf = x.astype(kFloat64).ravel().copy();
  const int64_t rows = xf.size();
  if (n < 0) n = rows;
  ndarray out = zeros({rows, n}, kFloat64);
  for (int64_t i = 0; i < rows; ++i) {
    const double xi = xf.item<double>({i});
    for (int64_t j = 0; j < n; ++j) {
      const int64_t p = increasing ? j : (n - 1 - j);
      out.set_item<double>({i, j}, std::pow(xi, static_cast<double>(p)));
    }
  }
  return out;
}

ndarray logspace(double start, double stop, int64_t num, bool endpoint, double base, DType dtype) {
  ndarray exps = linspace(start, stop, num, endpoint, kFloat64);
  ndarray b = full({}, base, kFloat64);
  return power(b, exps).astype(dtype);
}

ndarray geomspace(double start, double stop, int64_t num, bool endpoint, DType dtype) {
  const double sign = start < 0 ? -1.0 : 1.0;  // numpy requires start/stop same sign
  ndarray exps = linspace(std::log(std::abs(start)), std::log(std::abs(stop)), num, endpoint, kFloat64);
  ndarray vals = exp(exps);
  if (sign < 0) vals = multiply(vals, full(vals.shape(), -1.0, kFloat64));
  return vals.astype(dtype);
}

ndarray fromfunction(const std::function<double(const std::vector<int64_t>&)>& fn,
                     const Shape& shape, DType dtype) {
  ndarray out = zeros(shape, kFloat64);
  const int64_t total = out.size();
  const int64_t nd = static_cast<int64_t>(shape.size());
  std::vector<int64_t> idx(nd, 0);
  double* p = total ? out.typed_data<double>() : nullptr;
  for (int64_t flat = 0; flat < total; ++flat) {
    p[ravel_index(idx, shape)] = fn(idx);
    for (int64_t d = nd - 1; d >= 0; --d) {  // odometer increment
      if (++idx[d] < shape[d]) break;
      idx[d] = 0;
    }
  }
  return out.astype(dtype);
}

}  // namespace numpp
