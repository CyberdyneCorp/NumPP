#include "numpp/random/random.hpp"

namespace numpp {
namespace random {

ndarray Generator::random_raw(int64_t n) {
  ndarray out(Shape{n}, kUInt64, Order::C);
  uint64_t* p = out.typed_data<uint64_t>();
  for (int64_t i = 0; i < n; ++i) p[i] = bit_.next64();
  return out;
}

ndarray Generator::random(const Shape& shape) {
  ndarray out(shape, kFloat64, Order::C);
  const int64_t n = out.size();
  double* p = out.typed_data<double>();
  for (int64_t i = 0; i < n; ++i) p[i] = bit_.next_double();
  return out;
}

}  // namespace random
}  // namespace numpp
