#include "numpp/linalg/array_api.hpp"

#include <limits>

#include "numpp/core/error.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/linalg/linalg.hpp"
#include "numpp/umath/ufunc.hpp"

namespace numpp {
namespace linalg {

ndarray matrix_transpose(const ndarray& a) {
  if (a.ndim() < 2) throw value_error("matrix_transpose: array must be at least 2-D");
  return a.swapaxes(a.ndim() - 2, a.ndim() - 1);
}

ndarray permute_dims(const ndarray& a, const std::vector<int64_t>& axes) {
  return a.transpose(axes);
}

ndarray vecdot(const ndarray& a, const ndarray& b, int64_t axis) {
  // numpy.vecdot conjugates the first argument: sum(conj(a) * b) over `axis`.
  ndarray prod = multiply(conj(a), b);
  return sum(prod, normalize_axis(axis, prod.ndim()));
}

ndarray vector_norm(const ndarray& x, std::optional<int64_t> axis, double ord) {
  ndarray v = axis ? x : x.ravel();
  const int64_t ax = axis ? normalize_axis(*axis, v.ndim()) : 0;
  ndarray av = absolute(v);  // real magnitudes (handles complex input)
  const double inf = std::numeric_limits<double>::infinity();
  if (ord == 2.0) return sqrt(sum(square(av), ax));
  if (ord == 1.0) return sum(av, ax);
  if (ord == inf) return amax(av, ax);
  if (ord == -inf) return amin(av, ax);
  if (ord == 0.0)
    return sum(not_equal(v, scalar_like(0.0, v.dtype(), false)), ax).astype(kFloat64);
  ndarray s = sum(power(av, scalar_like(ord, kFloat64, true)), ax);
  return power(s, scalar_like(1.0 / ord, kFloat64, true));
}

ndarray matrix_norm(const ndarray& x, const std::string& ord) {
  return norm(x, ord);
}

}  // namespace linalg
}  // namespace numpp
