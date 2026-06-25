#pragma once

#include <cstdint>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"
#include "numpp/random/bitgen.hpp"

namespace numpp {
namespace random {

// numpy.random.Generator over a PCG64 BitGenerator (bit-exact stream).
class NUMPP_API Generator {
 public:
  explicit Generator(uint64_t seed) : bit_(seed) {}

  uint64_t next_raw() { return bit_.next64(); }
  double next_double() { return bit_.next_double(); }

  ndarray random_raw(int64_t n);                 // uint64 array (raw bit stream)
  ndarray random(const Shape& shape);            // float64 in [0, 1)
  ndarray random(int64_t n) { return random(Shape{n}); }

 private:
  PCG64 bit_;
};

// numpy.random.default_rng(seed) equivalent.
inline Generator default_rng(uint64_t seed) { return Generator(seed); }

}  // namespace random
}  // namespace numpp
