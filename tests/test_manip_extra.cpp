#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("block 2x2 vs numpy") {
  ndarray A = arange(0., 4., 1., kFloat64).reshape({2, 2});
  ndarray B = arange(10., 14., 1., kFloat64).reshape({2, 2});
  ndarray C = arange(20., 24., 1., kFloat64).reshape({2, 2});
  ndarray D = arange(30., 34., 1., kFloat64).reshape({2, 2});
  auto o = npt::oracle(
      "A=np.arange(4.).reshape(2,2);B=np.arange(10,14.).reshape(2,2);"
      "C=np.arange(20,24.).reshape(2,2);D=np.arange(30,34.).reshape(2,2);"
      "a=np.block([[A,B],[C,D]])");
  if (o) {
    ndarray r = block({{A, B}, {C, D}});
    CHECK((r.shape() == Shape{4, 4}));
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("block uneven column widths vs numpy") {
  ndarray A = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray B = arange(100., 102., 1., kFloat64).reshape({2, 1});
  ndarray C = arange(200., 208., 1., kFloat64).reshape({2, 4});
  auto o = npt::oracle(
      "A=np.arange(6.).reshape(2,3);B=np.arange(100,102.).reshape(2,1);"
      "C=np.arange(200,208.).reshape(2,4);a=np.block([[A,B],[C]])");
  if (o) {
    ndarray r = block({{A, B}, {C}});
    CHECK((r.shape() == Shape{4, 4}));
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("dsplit even vs numpy") {
  ndarray x = arange(0., 24., 1., kFloat64).reshape({2, 3, 4});
  auto o0 = npt::oracle("a=np.dsplit(np.arange(24.).reshape(2,3,4),2)[0]");
  auto o1 = npt::oracle("a=np.dsplit(np.arange(24.).reshape(2,3,4),2)[1]");
  if (o0 && o1) {
    auto parts = dsplit(x, 2);
    CHECK(parts.size() == 2);
    CHECK((parts[0].shape() == Shape{2, 3, 2}));
    CHECK(allclose(parts[0], *o0, 1e-9, 1e-12, true));
    CHECK(allclose(parts[1], *o1, 1e-9, 1e-12, true));
  }
}

TEST_CASE("dsplit requires 3d") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  CHECK_THROWS_AS(dsplit(x, 2), value_error);
}

TEST_CASE("trim_zeros fb vs numpy") {
  ndarray x(Shape{6}, kFloat64);
  double vals[6] = {0., 0., 1., 2., 0., 0.};
  for (int64_t i = 0; i < 6; ++i) x.set_item<double>({i}, vals[i]);
  auto o = npt::oracle("a=np.trim_zeros(np.array([0,0,1,2,0,0.]))");
  if (o) CHECK(allclose(trim_zeros(x, "fb"), *o, 1e-9, 1e-12, true));
}

TEST_CASE("trim_zeros front only vs numpy") {
  ndarray x(Shape{6}, kFloat64);
  double vals[6] = {0., 0., 1., 2., 0., 0.};
  for (int64_t i = 0; i < 6; ++i) x.set_item<double>({i}, vals[i]);
  auto of = npt::oracle("a=np.trim_zeros(np.array([0,0,1,2,0,0.]),'f')");
  auto ob = npt::oracle("a=np.trim_zeros(np.array([0,0,1,2,0,0.]),'b')");
  if (of) CHECK(allclose(trim_zeros(x, "f"), *of, 1e-9, 1e-12, true));
  if (ob) CHECK(allclose(trim_zeros(x, "b"), *ob, 1e-9, 1e-12, true));
}

TEST_CASE("trim_zeros all zero is empty") {
  ndarray x = zeros({4}, kFloat64);
  CHECK(trim_zeros(x, "fb").size() == 0);
}

TEST_CASE("rollaxis 2,0 vs numpy") {
  ndarray x = arange(0., 24., 1., kFloat64).reshape({2, 3, 4});
  auto o = npt::oracle("a=np.rollaxis(np.arange(24.).reshape(2,3,4),2,0)");
  if (o) {
    ndarray r = rollaxis(x, 2, 0);
    CHECK((r.shape() == Shape{4, 2, 3}));
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("rollaxis 1,3 vs numpy") {
  ndarray x = arange(0., 24., 1., kFloat64).reshape({2, 3, 4});
  auto o = npt::oracle("a=np.rollaxis(np.arange(24.).reshape(2,3,4),1,3)");
  if (o) {
    ndarray r = rollaxis(x, 1, 3);
    CHECK((r.shape() == Shape{2, 4, 3}));
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}

TEST_CASE("rollaxis negative axis vs numpy") {
  ndarray x = arange(0., 24., 1., kFloat64).reshape({2, 3, 4});
  auto o = npt::oracle("a=np.rollaxis(np.arange(24.).reshape(2,3,4),-1,1)");
  if (o) CHECK(allclose(rollaxis(x, -1, 1), *o, 1e-9, 1e-12, true));
}
