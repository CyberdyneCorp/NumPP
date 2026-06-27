#include "numpp/lib/iteration.hpp"

#include <complex>
#include <cstring>
#include <type_traits>

#include "numpp/core/dtype.hpp"
#include "numpp/core/error.hpp"
#include "numpp/core/shape.hpp"

namespace numpp {

namespace {

// Advance a C-order multi-index in place; returns false past the last index.
bool advance_c(std::vector<int64_t>& idx, const Shape& shape) {
  for (int64_t ax = static_cast<int64_t>(shape.size()) - 1; ax >= 0; --ax) {
    if (++idx[ax] < shape[ax]) return true;
    idx[ax] = 0;
  }
  return false;
}

// Read the element at `idx` as a double (real numeric dtypes only). Uses strided
// element access, so it is correct for non-contiguous / broadcasted views.
double read_as_double(const ndarray& a, const std::vector<int64_t>& idx) {
  const char* p = a.element_ptr(idx);
  return visit_dtype(a.dtype().id(), [&](auto tag) -> double {
    using T = typename decltype(tag)::type;
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
      throw type_error("ndenumerate/nditer: complex dtype not supported");
    } else if constexpr (std::is_same_v<T, half>) {
      T v;
      std::memcpy(&v, p, sizeof(T));
      return static_cast<double>(static_cast<float>(v));
    } else {
      T v;
      std::memcpy(&v, p, sizeof(T));
      return static_cast<double>(v);
    }
  });
}

}  // namespace

std::vector<std::vector<int64_t>> ndindex(const Shape& shape) {
  std::vector<std::vector<int64_t>> out;
  if (shape_size(shape) == 0) return out;  // a zero-length axis yields no indices
  out.reserve(static_cast<size_t>(shape_size(shape)));
  std::vector<int64_t> idx(shape.size(), 0);
  do {
    out.push_back(idx);
  } while (advance_c(idx, shape));
  return out;
}

std::vector<std::pair<std::vector<int64_t>, double>> ndenumerate(const ndarray& a) {
  if (a.dtype().is_complex())
    throw type_error("ndenumerate: complex dtype not supported");
  std::vector<std::pair<std::vector<int64_t>, double>> out;
  const auto idxs = ndindex(a.shape());
  out.reserve(idxs.size());
  for (const auto& idx : idxs) out.emplace_back(idx, read_as_double(a, idx));
  return out;
}

ndarray nditer(const ndarray& a) {
  if (a.dtype().is_complex())
    throw type_error("nditer: complex dtype not supported");
  const auto idxs = ndindex(a.shape());
  ndarray out(Shape{static_cast<int64_t>(idxs.size())}, kFloat64, Order::C);
  double* dst = out.typed_data<double>();
  for (size_t i = 0; i < idxs.size(); ++i) dst[i] = read_as_double(a, idxs[i]);
  return out;
}

}  // namespace numpp
