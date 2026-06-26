#include "numpp/lib/stride_tricks2.hpp"

#include "numpp/core/creation.hpp"
#include "numpp/core/shape.hpp"
#include "numpp/umath/ufunc.hpp"
#include "numpp/core/error.hpp"
#include <functional>
#include <vector>


namespace numpp {

ndarray vectorize(const std::function<double(double)>& func, const ndarray& a) {
  // Apply func elementwise over a float64 contiguous copy; preserve shape.
  ndarray src = a.ascontiguousarray().astype(kFloat64);
  ndarray out(src.shape(), kFloat64, Order::C);
  const int64_t n = src.size();
  const double* in = src.typed_data<double>();
  double* dst = out.typed_data<double>();
  for (int64_t i = 0; i < n; ++i) dst[i] = func(in[i]);
  return out;
}

ndarray apply_over_axes(const std::function<ndarray(const ndarray&, int64_t)>& func,
                        const ndarray& a, const std::vector<int64_t>& axes) {
  ndarray res = a;
  for (int64_t ax : axes) {
    const int64_t before = res.ndim();
    ndarray next = func(res, ax);
    // numpy keeps the reduced axis; if func dropped it, re-insert a length-1 axis.
    if (next.ndim() == before - 1) {
      const int64_t norm = normalize_axis(ax, before);
      next = next.expand_dims(norm);
    }
    res = next;
  }
  return res;
}

}  // namespace numpp
