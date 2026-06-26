#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("sliding_window_view 1d vs numpy") {
  auto o = npt::oracle("a=np.lib.stride_tricks.sliding_window_view(np.arange(10.),3)");
  if (o) {
    ndarray x = arange(0.0, 10.0, 1.0, kFloat64);
    CHECK((allclose(sliding_window_view(x, {3}), *o, 1e-9, 1e-12, true)));
  }
}

TEST_CASE("sliding_window_view 2d vs numpy") {
  auto o = npt::oracle("a=np.lib.stride_tricks.sliding_window_view(np.arange(12.).reshape(3,4),(2,2))");
  if (o) {
    ndarray x = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
    CHECK((allclose(sliding_window_view(x, {2, 2}), *o, 1e-9, 1e-12, true)));
  }
}

TEST_CASE("sliding_window_view single trailing axis of 2d vs numpy") {
  auto o = npt::oracle("a=np.lib.stride_tricks.sliding_window_view(np.arange(12.).reshape(3,4),3,axis=1)");
  if (o) {
    ndarray x = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
    CHECK((allclose(sliding_window_view(x, {3}), *o, 1e-9, 1e-12, true)));
  }
}

TEST_CASE("as_strided overlapping windows vs numpy") {
  auto o = npt::oracle("x=np.arange(5.); a=np.lib.stride_tricks.as_strided(x,(3,3),(8,8))");
  if (o) {
    ndarray x = arange(0.0, 5.0, 1.0, kFloat64);
    CHECK((allclose(as_strided(x, {3, 3}, {8, 8}), *o, 1e-9, 1e-12, true)));
  }
}

TEST_CASE("as_strided transpose via strides vs numpy") {
  auto o = npt::oracle("x=np.arange(6.).reshape(2,3); a=np.lib.stride_tricks.as_strided(x,(3,2),(8,24))");
  if (o) {
    ndarray x = arange(0.0, 6.0, 1.0, kFloat64).reshape({2, 3});
    CHECK((allclose(as_strided(x, {3, 2}, {8, 24}), *o, 1e-9, 1e-12, true)));
  }
}

TEST_CASE("piecewise vs numpy") {
  auto o = npt::oracle("x=np.linspace(-2,2,5); a=np.piecewise(x,[x<0,x>=0],[lambda v:-v, lambda v:v*v])");
  if (o) {
    ndarray x = linspace(-2.0, 2.0, 5, true, kFloat64);
    ndarray c0(Shape{5}, kBool), c1(Shape{5}, kBool);
    for (int64_t i = 0; i < 5; ++i) {
      const double v = x.item<double>({i});
      c0.set_item<bool>({i}, v < 0);
      c1.set_item<bool>({i}, v >= 0);
    }
    std::vector<ndarray> conds{c0, c1};
    std::vector<std::function<double(double)>> funcs{[](double v) { return -v; },
                                                     [](double v) { return v * v; }};
    CHECK(allclose(piecewise(x, conds, funcs), *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("piecewise with default vs numpy") {
  auto o = npt::oracle("x=np.linspace(-3,3,7); a=np.piecewise(x,[x<-1,x>1],[lambda v:-1.0, lambda v:1.0, lambda v:0.5])");
  if (o) {
    ndarray x = linspace(-3.0, 3.0, 7, true, kFloat64);
    ndarray c0(Shape{7}, kBool), c1(Shape{7}, kBool);
    for (int64_t i = 0; i < 7; ++i) {
      const double v = x.item<double>({i});
      c0.set_item<bool>({i}, v < -1);
      c1.set_item<bool>({i}, v > 1);
    }
    std::vector<ndarray> conds{c0, c1};
    std::vector<std::function<double(double)>> funcs{
        [](double) { return -1.0; }, [](double) { return 1.0; }, [](double) { return 0.5; }};
    CHECK(allclose(piecewise(x, conds, funcs), *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("apply_along_axis cumsum axis=1 vs numpy") {
  auto o = npt::oracle("a=np.apply_along_axis(np.cumsum,1,np.arange(12.).reshape(3,4))");
  if (o) {
    ndarray x = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
    auto f = [](const ndarray& s) { return cumsum(s); };
    CHECK(allclose(apply_along_axis(f, 1, x), *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("apply_along_axis cumsum axis=0 vs numpy") {
  auto o = npt::oracle("a=np.apply_along_axis(np.cumsum,0,np.arange(12.).reshape(3,4))");
  if (o) {
    ndarray x = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
    auto f = [](const ndarray& s) { return cumsum(s); };
    CHECK(allclose(apply_along_axis(f, 0, x), *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("apply_along_axis scalar reduction collapses axis vs numpy") {
  auto o = npt::oracle("a=np.apply_along_axis(np.sum,1,np.arange(12.).reshape(3,4))");
  if (o) {
    ndarray x = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
    auto f = [](const ndarray& s) { return sum(s); };
    CHECK(allclose(apply_along_axis(f, 1, x), *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("polyvander vs numpy") {
  auto o = npt::oracle("a=np.polynomial.polynomial.polyvander([0,1,2,3.],3)");
  if (o) {
    ndarray x = arange(0.0, 4.0, 1.0, kFloat64);
    CHECK(allclose(polyvander(x, 3), *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("polycompanion vs numpy") {
  auto o = npt::oracle("a=np.polynomial.polynomial.polycompanion([-6,11,-6,1.])");
  if (o) {
    ndarray c(Shape{4}, kFloat64);
    c.set_item<double>({0}, -6.0);
    c.set_item<double>({1}, 11.0);
    c.set_item<double>({2}, -6.0);
    c.set_item<double>({3}, 1.0);
    CHECK(allclose(polycompanion(c), *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("mask_indices tril rows vs numpy") {
  auto o = npt::oracle("a=np.mask_indices(4,np.tril)[0]");
  if (o) {
    auto r = mask_indices(4, "tril");
    CHECK(allclose(r[0], *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("mask_indices tril cols vs numpy") {
  auto o = npt::oracle("a=np.mask_indices(4,np.tril)[1]");
  if (o) {
    auto r = mask_indices(4, "tril");
    CHECK(allclose(r[1], *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("mask_indices triu rows vs numpy") {
  auto o = npt::oracle("a=np.mask_indices(5,np.triu)[0]");
  if (o) {
    auto r = mask_indices(5, "triu");
    CHECK(allclose(r[0], *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("mask_indices triu k=1 cols vs numpy") {
  auto o = npt::oracle("a=np.mask_indices(4,np.triu,1)[1]");
  if (o) {
    auto r = mask_indices(4, "triu", 1);
    CHECK(allclose(r[1], *o, 1e-9, 1e-12, true));
  }
}
