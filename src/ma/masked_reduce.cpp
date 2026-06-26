#include "numpp/ma/masked_extra.hpp"

#include "numpp/ma/masked.hpp"
#include "numpp/umath/ufunc.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/core/error.hpp"
#include <algorithm>
#include <cmath>
#include <vector>


namespace numpp {

namespace ma {

namespace {

// Move 'axis' to the last position via transpose, returning a contiguous copy.
// For an N-d array this builds perm = [all dims except axis, axis].
inline ndarray move_axis_last_contig(const ndarray& a, int64_t axis, bool as_bool) {
  const int64_t nd = a.ndim();
  std::vector<int64_t> perm;
  perm.reserve(static_cast<size_t>(nd));
  for (int64_t d = 0; d < nd; ++d)
    if (d != axis) perm.push_back(d);
  perm.push_back(axis);
  ndarray t = a.transpose(perm);
  ndarray casted = as_bool ? t.astype(kBool) : t.astype(kFloat64);
  return casted.ascontiguousarray();
}

// Shape with 'axis' removed (preserving the order of remaining dims).
inline Shape reduced_shape(const ndarray& a, int64_t axis) {
  const Shape& s = a.shape();
  Shape out;
  for (int64_t d = 0; d < a.ndim(); ++d)
    if (d != axis) out.push_back(s[static_cast<size_t>(d)]);
  return out;
}

enum class ReduceOp { Sum, Prod, Mean, Max, Min };

// Generic reduction of the unmasked elements along the last (length-L) axis.
MaskedArray reduce_axis(const MaskedArray& m, int64_t axis, ReduceOp op) {
  const int64_t a = normalize_axis(axis, m.data.ndim());
  const int64_t L = m.data.shape()[static_cast<size_t>(a)];
  ndarray dC = move_axis_last_contig(m.data, a, /*as_bool=*/false);
  ndarray mC = move_axis_last_contig(m.mask, a, /*as_bool=*/true);
  const double* dp = dC.typed_data<double>();
  const bool* mp = mC.typed_data<bool>();

  Shape oshape = reduced_shape(m.data, a);
  const int64_t rows = (L > 0) ? (dC.size() / L) : dC.size();

  ndarray outdata(oshape, kFloat64);
  ndarray outmask(oshape, kBool);
  // outdata / outmask are freshly allocated C-order => contiguous flat access.
  ndarray odf = outdata.ravel(Order::C);
  ndarray omf = outmask.ravel(Order::C);
  double* odp = odf.typed_data<double>();
  bool* omp = omf.typed_data<bool>();

  for (int64_t r = 0; r < rows; ++r) {
    const double* row = dp + r * L;
    const bool* rmask = mp + r * L;
    double acc = (op == ReduceOp::Prod) ? 1.0 : 0.0;
    double best = 0.0;
    int64_t cnt = 0;
    for (int64_t j = 0; j < L; ++j) {
      if (rmask[j]) continue;
      const double v = row[j];
      switch (op) {
        case ReduceOp::Sum:
        case ReduceOp::Mean: acc += v; break;
        case ReduceOp::Prod: acc *= v; break;
        case ReduceOp::Max: if (cnt == 0 || v > best) best = v; break;
        case ReduceOp::Min: if (cnt == 0 || v < best) best = v; break;
      }
      ++cnt;
    }
    const bool empty = (cnt == 0);
    omp[r] = empty;
    switch (op) {
      case ReduceOp::Sum: odp[r] = acc; break;
      case ReduceOp::Prod: odp[r] = acc; break;
      case ReduceOp::Mean:
        odp[r] = empty ? std::nan("") : acc / static_cast<double>(cnt);
        break;
      case ReduceOp::Max:
      case ReduceOp::Min: odp[r] = empty ? std::nan("") : best; break;
    }
  }
  return MaskedArray{outdata, outmask};
}

}  // namespace

MaskedArray sum_axis(const MaskedArray& m, int64_t axis) {
  return reduce_axis(m, axis, ReduceOp::Sum);
}

MaskedArray prod_axis(const MaskedArray& m, int64_t axis) {
  return reduce_axis(m, axis, ReduceOp::Prod);
}

MaskedArray mean_axis(const MaskedArray& m, int64_t axis) {
  return reduce_axis(m, axis, ReduceOp::Mean);
}

MaskedArray max_axis(const MaskedArray& m, int64_t axis) {
  return reduce_axis(m, axis, ReduceOp::Max);
}

MaskedArray min_axis(const MaskedArray& m, int64_t axis) {
  return reduce_axis(m, axis, ReduceOp::Min);
}

ndarray count_axis(const MaskedArray& m, int64_t axis) {
  const int64_t a = normalize_axis(axis, m.mask.ndim());
  const int64_t L = m.mask.shape()[static_cast<size_t>(a)];
  ndarray mC = move_axis_last_contig(m.mask, a, /*as_bool=*/true);
  const bool* mp = mC.typed_data<bool>();

  Shape oshape = reduced_shape(m.mask, a);
  const int64_t rows = (L > 0) ? (mC.size() / L) : mC.size();

  ndarray out(oshape, kInt64);
  ndarray of = out.ravel(Order::C);
  int64_t* op = of.typed_data<int64_t>();
  for (int64_t r = 0; r < rows; ++r) {
    const bool* rmask = mp + r * L;
    int64_t cnt = 0;
    for (int64_t j = 0; j < L; ++j)
      if (!rmask[j]) ++cnt;
    op[r] = cnt;
  }
  return out;
}

}  // namespace ma

}  // namespace numpp
