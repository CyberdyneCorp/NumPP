#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("vectorize square plus one vs numpy") {
  auto o = npt::oracle("a=np.vectorize(lambda v: v*v+1)(np.arange(6.))");
  if (o) {
    auto x = arange(0.0, 6.0, 1.0, kFloat64);
    CHECK((allclose(vectorize([](double v) { return v * v + 1; }, x), *o, 1e-9, 1e-12, true)));
  }
}

TEST_CASE("vectorize negate vs numpy") {
  auto o = npt::oracle("a=np.vectorize(lambda v: -v)(np.arange(10.))");
  if (o) {
    auto x = arange(0.0, 10.0, 1.0, kFloat64);
    CHECK((allclose(vectorize([](double v) { return -v; }, x), *o, 1e-9, 1e-12, true)));
  }
}

TEST_CASE("vectorize preserves 2d shape vs numpy") {
  auto o = npt::oracle("a=np.vectorize(lambda v: 2*v+3)(np.arange(12.).reshape(3,4))");
  if (o) {
    auto x = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
    auto r = vectorize([](double v) { return 2 * v + 3; }, x);
    CHECK((r.shape() == Shape{3, 4}));
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("vectorize casts integer input to float vs numpy") {
  auto o = npt::oracle("a=np.vectorize(lambda v: v*v)(np.arange(5))");
  if (o) {
    auto x = arange(0.0, 5.0, 1.0, kInt64);
    CHECK((allclose(vectorize([](double v) { return v * v; }, x), *o, 1e-9, 1e-12, true)));
  }
}

TEST_CASE("apply_over_axes sum over two axes vs numpy") {
  auto o = npt::oracle("a=np.apply_over_axes(np.sum, np.arange(24.).reshape(2,3,4), [0,2])");
  if (o) {
    auto X = arange(0.0, 24.0, 1.0, kFloat64).reshape({2, 3, 4});
    auto r = apply_over_axes(
        [](const ndarray& x, int64_t ax) { return sum(x, ax, true); }, X, {0, 2});
    CHECK((r.shape() == Shape{1, 3, 1}));
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("apply_over_axes single axis vs numpy") {
  auto o = npt::oracle("a=np.apply_over_axes(np.sum, np.arange(24.).reshape(2,3,4), [1])");
  if (o) {
    auto X = arange(0.0, 24.0, 1.0, kFloat64).reshape({2, 3, 4});
    auto r = apply_over_axes(
        [](const ndarray& x, int64_t ax) { return sum(x, ax, true); }, X, {1});
    CHECK((r.shape() == Shape{2, 1, 4}));
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("apply_over_axes all axes vs numpy") {
  auto o = npt::oracle("a=np.apply_over_axes(np.sum, np.arange(24.).reshape(2,3,4), [0,1,2])");
  if (o) {
    auto X = arange(0.0, 24.0, 1.0, kFloat64).reshape({2, 3, 4});
    auto r = apply_over_axes(
        [](const ndarray& x, int64_t ax) { return sum(x, ax, true); }, X, {0, 1, 2});
    CHECK((r.shape() == Shape{1, 1, 1}));
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}
