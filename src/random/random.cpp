#include "numpp/random/random.hpp"

#include <cstring>
#include <vector>

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

// Lemire bounded uint in [0, range_excl). 32-bit buffered path for ranges that
// fit in uint32 (matches numpy), else 64-bit.
uint64_t Generator::bounded(uint64_t range_excl) {
  if (range_excl <= 1) return 0;
  if (range_excl <= 0x100000000ull) {
    const uint64_t thr = (0x100000000ull - range_excl) % range_excl;
    uint64_t m = static_cast<uint64_t>(bit_.next32()) * range_excl;
    uint64_t lo = m & 0xffffffffull;
    while (lo < thr) { m = static_cast<uint64_t>(bit_.next32()) * range_excl; lo = m & 0xffffffffull; }
    return m >> 32;
  }
  const uint64_t thr = (0ull - range_excl) % range_excl;
  u128 m = static_cast<u128>(bit_.next64()) * range_excl;
  uint64_t lo = static_cast<uint64_t>(m);
  while (lo < thr) { m = static_cast<u128>(bit_.next64()) * range_excl; lo = static_cast<uint64_t>(m); }
  return static_cast<uint64_t>(m >> 64);
}

// Masked rejection in [0, max_inclusive] (numpy random_interval, used by shuffle).
uint64_t Generator::interval(uint64_t mx) {
  if (mx == 0) return 0;
  uint64_t mask = mx;
  mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
  if (mx <= 0xffffffffull) { uint64_t v; do { v = bit_.next32() & mask; } while (v > mx); return v; }
  uint64_t v; do { v = bit_.next64() & mask; } while (v > mx); return v;
}

ndarray Generator::integers(int64_t low, int64_t high, const Shape& shape, bool endpoint) {
  const uint64_t range_excl = static_cast<uint64_t>(high - low) + (endpoint ? 1u : 0u);
  if (range_excl == 0) throw value_error("integers: low >= high");
  ndarray out(shape, kInt64, Order::C);
  const int64_t n = out.size();
  int64_t* p = out.typed_data<int64_t>();
  for (int64_t i = 0; i < n; ++i) p[i] = low + static_cast<int64_t>(bounded(range_excl));
  return out;
}

ndarray Generator::uniform(double low, double high, const Shape& shape) {
  ndarray out(shape, kFloat64, Order::C);
  const int64_t n = out.size();
  double* p = out.typed_data<double>();
  const double range = high - low;
  for (int64_t i = 0; i < n; ++i) p[i] = low + range * bit_.next_double();
  return out;
}

void Generator::shuffle(ndarray& a) {
  if (a.ndim() == 0) return;
  const int64_t n = a.shape()[0];
  const int64_t row = a.size() / (n ? n : 1) * a.itemsize();  // bytes per row along axis 0
  char* base = a.bytes();
  std::vector<char> tmp(row);
  for (int64_t i = n - 1; i > 0; --i) {
    int64_t j = static_cast<int64_t>(interval(static_cast<uint64_t>(i)));
    if (j != i) {
      std::memcpy(tmp.data(), base + i * row, row);
      std::memcpy(base + i * row, base + j * row, row);
      std::memcpy(base + j * row, tmp.data(), row);
    }
  }
}

ndarray Generator::permutation(int64_t n) {
  ndarray x(Shape{n}, kInt64, Order::C);
  int64_t* p = x.typed_data<int64_t>();
  for (int64_t i = 0; i < n; ++i) p[i] = i;
  shuffle(x);
  return x;
}
ndarray Generator::permutation(const ndarray& a) {
  ndarray c = a.copy();
  shuffle(c);
  return c;
}
ndarray Generator::choice(int64_t n, int64_t size, bool replace) {
  if (replace) return integers(0, n, Shape{size});
  if (size > n) throw value_error("choice: size larger than population without replacement");
  ndarray perm = permutation(n);
  return perm.index({IndexItem{Slice{0, size, 1}}}).copy();
}

}  // namespace random
}  // namespace numpp
