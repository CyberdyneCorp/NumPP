#include "numpp/construct/construct.hpp"

#include "numpp/core/creation.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/core/error.hpp"
#include <array>
#include <cstring>
#include <vector>


namespace numpp {

ndarray fromiter(const std::vector<double>& data, DType dtype) {
  const int64_t n = static_cast<int64_t>(data.size());
  ndarray tmp({n}, kFloat64, Order::C);
  double* p = n ? tmp.typed_data<double>() : nullptr;
  for (int64_t i = 0; i < n; ++i) p[i] = data[static_cast<size_t>(i)];
  return dtype == kFloat64 ? tmp : tmp.astype(dtype);
}

ndarray frombuffer(const std::string& buffer, DType dtype, int64_t count) {
  const int64_t isz = dtype.itemsize();
  const int64_t total = static_cast<int64_t>(buffer.size());
  const int64_t n = count < 0 ? total / isz : count;
  if (n < 0 || n * isz > total) throw value_error("frombuffer: buffer too small for requested count");
  ndarray out({n}, dtype, Order::C);
  if (n > 0) std::memcpy(out.bytes(), buffer.data(), static_cast<size_t>(n * isz));
  return out;
}

std::vector<ndarray> broadcast_arrays(const std::vector<ndarray>& arrays) {
  std::vector<ndarray> out;
  if (arrays.empty()) return out;
  Shape common = arrays[0].shape();
  for (size_t i = 1; i < arrays.size(); ++i) common = broadcast_shapes(common, arrays[i].shape());
  out.reserve(arrays.size());
  for (const ndarray& a : arrays) out.push_back(a.broadcast_to(common).copy());
  return out;
}

std::vector<ndarray> meshgrid_sparse(const std::vector<ndarray>& xi, bool indexing_xy) {
  const int64_t n = static_cast<int64_t>(xi.size());
  std::vector<ndarray> out;
  out.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    Shape rshape(static_cast<size_t>(n), 1);
    rshape[static_cast<size_t>(i)] = xi[static_cast<size_t>(i)].size();
    ndarray g = xi[static_cast<size_t>(i)].reshape(rshape);
    if (indexing_xy && n >= 2) {
      std::vector<int64_t> perm(static_cast<size_t>(n));
      for (int64_t d = 0; d < n; ++d) perm[static_cast<size_t>(d)] = d;
      perm[0] = 1;
      perm[1] = 0;
      g = g.transpose(perm).ascontiguousarray();
    } else {
      g = g.copy();
    }
    out.push_back(g);
  }
  return out;
}

namespace {

// Build the 1-D coordinate vectors (one per range) used by mgrid/ogrid.
std::vector<ndarray> grid_axes(const std::vector<std::array<double, 3>>& ranges) {
  std::vector<ndarray> axes;
  axes.reserve(ranges.size());
  for (const std::array<double, 3>& r : ranges)
    axes.push_back(arange(r[0], r[1], r[2], kFloat64));
  return axes;
}

}  // namespace

std::vector<ndarray> mgrid(const std::vector<std::array<double, 3>>& ranges) {
  std::vector<ndarray> axes = grid_axes(ranges);
  const int64_t n = static_cast<int64_t>(axes.size());
  Shape full(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) full[static_cast<size_t>(i)] = axes[static_cast<size_t>(i)].size();
  std::vector<ndarray> out;
  out.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    Shape rshape(static_cast<size_t>(n), 1);
    rshape[static_cast<size_t>(i)] = axes[static_cast<size_t>(i)].size();
    out.push_back(axes[static_cast<size_t>(i)].reshape(rshape).broadcast_to(full).copy());
  }
  return out;
}

std::vector<ndarray> ogrid(const std::vector<std::array<double, 3>>& ranges) {
  std::vector<ndarray> axes = grid_axes(ranges);
  const int64_t n = static_cast<int64_t>(axes.size());
  std::vector<ndarray> out;
  out.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    Shape rshape(static_cast<size_t>(n), 1);
    rshape[static_cast<size_t>(i)] = axes[static_cast<size_t>(i)].size();
    out.push_back(axes[static_cast<size_t>(i)].reshape(rshape).copy());
  }
  return out;
}

}  // namespace numpp
