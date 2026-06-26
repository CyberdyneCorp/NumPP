#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <string>

using namespace numpp;

// Bit-exact ziggurat standard_normal / standard_exponential / normal, and
// choice(replace=False), vs numpy.random.default_rng(seed).

TEST_CASE("ziggurat standard_normal bit-exact vs numpy") {
  for (uint64_t seed : {12345ull, 0ull, 7ull}) {
    auto g = random::default_rng(seed);
    ndarray r = g.standard_normal(Shape{50});
    auto o = npt::oracle("a=np.random.default_rng(" + std::to_string(seed) + ").standard_normal(50)");
    if (o) CHECK(allclose(r, *o, 0.0, 0.0, true));  // bit-exact
  }
}

TEST_CASE("ziggurat standard_exponential bit-exact vs numpy") {
  for (uint64_t seed : {12345ull, 777ull, 3ull}) {
    auto g = random::default_rng(seed);
    ndarray r = g.standard_exponential(Shape{50});
    auto o = npt::oracle("a=np.random.default_rng(" + std::to_string(seed) + ").standard_exponential(50)");
    if (o) CHECK(allclose(r, *o, 0.0, 0.0, true));
  }
}

TEST_CASE("normal bit-exact vs numpy") {
  auto g = random::default_rng(42);
  ndarray r = g.normal(2.5, 1.5, Shape{40});
  auto o = npt::oracle("a=np.random.default_rng(42).normal(2.5,1.5,40)");
  if (o) CHECK(allclose(r, *o, 0.0, 0.0, true));
}

TEST_CASE("choice replace=False bit-exact vs numpy (Floyd path)") {
  for (uint64_t seed : {12345ull, 7ull, 999ull}) {
    auto g = random::default_rng(seed);
    ndarray c = g.choice(20, 6, false);
    auto o = npt::oracle("a=np.random.default_rng(" + std::to_string(seed) +
                         ").choice(20,6,replace=False)");
    if (o) CHECK(allclose(c, *o, 0.0, 0.0, true));
  }
}

TEST_CASE("choice replace=False bit-exact vs numpy (tail-shuffle path)") {
  // pop > 10000 and size > pop/50 -> numpy's _shuffle_int tail branch.
  auto g = random::default_rng(2024);
  ndarray c = g.choice(20000, 1000, false);
  auto o = npt::oracle("a=np.random.default_rng(2024).choice(20000,1000,replace=False)");
  if (o) CHECK(allclose(c, *o, 0.0, 0.0, true));
}
