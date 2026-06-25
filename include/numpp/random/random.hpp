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

  // Bounded integers in [low, high) (or [low, high] if endpoint), int64 output.
  ndarray integers(int64_t low, int64_t high, const Shape& shape, bool endpoint = false);
  ndarray integers(int64_t low, int64_t high, int64_t n, bool endpoint = false) { return integers(low, high, Shape{n}, endpoint); }
  ndarray integers(int64_t high, int64_t n) { return integers(0, high, Shape{n}, false); }

  ndarray uniform(double low, double high, const Shape& shape);
  ndarray uniform(double low, double high, int64_t n) { return uniform(low, high, Shape{n}); }

  ndarray permutation(int64_t n);                // shuffled arange(n)
  ndarray permutation(const ndarray& a);         // shuffled copy (along axis 0)
  void shuffle(ndarray& a);                      // in place, along axis 0
  ndarray choice(int64_t n, int64_t size, bool replace = true);

  // Distributions (statistically correct; not bit-exact with numpy — issue #8).
  double next_gauss();                           // single standard normal
  double next_exponential();                     // single standard exponential
  double next_std_gamma(double shape);
  ndarray standard_normal(const Shape& shape);
  ndarray standard_normal(int64_t n) { return standard_normal(Shape{n}); }
  ndarray standard_exponential(const Shape& shape);
  ndarray normal(double loc, double scale, const Shape& shape);
  ndarray normal(double loc, double scale, int64_t n) { return normal(loc, scale, Shape{n}); }
  ndarray exponential(double scale, const Shape& shape);
  ndarray gamma(double shape_k, double scale, const Shape& size);
  ndarray beta(double a, double b, const Shape& size);
  ndarray chisquare(double df, const Shape& size);
  ndarray poisson(double lam, const Shape& size);
  ndarray binomial(int64_t n, double p, const Shape& size);

  // Raw bounded draws (exposed for reuse/testing).
  uint64_t bounded(uint64_t range_excl);         // Lemire, in [0, range_excl)
  uint64_t interval(uint64_t max_inclusive);     // masked, in [0, max_inclusive]

 private:
  PCG64 bit_;
  bool has_gauss_ = false;
  double gauss_ = 0.0;
};

// numpy.random.default_rng(seed) equivalent.
inline Generator default_rng(uint64_t seed) { return Generator(seed); }

}  // namespace random
}  // namespace numpp
