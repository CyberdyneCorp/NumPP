#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <vector>

using namespace numpp;

namespace {
ndarray ints(const std::vector<int64_t>& v, const Shape& shape) {
  ndarray a(shape, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i)
    a.typed_data<int64_t>()[i] = v[i];
  return a;
}
}  // namespace

// ============================= take =============================
TEST_CASE("take flat vs numpy") {
  ndarray a = arange(0., 10., 1., kFloat64);
  ndarray idx = ints({1, 3, 5, 8}, {4});
  auto o = npt::oracle("a=np.take(np.arange(10.),[1,3,5,8])");
  if (o) CHECK(allclose(take(a, idx), *o, 1e-9, 1e-12, true));
}
TEST_CASE("take flat negative index vs numpy") {
  ndarray a = arange(0., 10., 1., kFloat64);
  ndarray idx = ints({-1, -2}, {2});
  auto o = npt::oracle("a=np.take(np.arange(10.),[-1,-2])");
  if (o) CHECK(allclose(take(a, idx), *o, 1e-9, 1e-12, true));
}
TEST_CASE("take axis1 vs numpy") {
  ndarray m = arange(0., 12., 1., kFloat64).reshape({3, 4});
  ndarray idx = ints({0, 2}, {2});
  auto o = npt::oracle("a=np.take(np.arange(12.).reshape(3,4),[0,2],axis=1)");
  if (o) CHECK(allclose(take(m, idx, 1), *o, 1e-9, 1e-12, true));
}

// ============================= take_along_axis =============================
TEST_CASE("take_along_axis axis1 vs numpy") {
  ndarray m = arange(0., 12., 1., kFloat64).reshape({3, 4});
  ndarray idx = ints({3, 0, 1, 2, 0, 3, 1, 2, 2, 1, 0, 3}, {3, 4});
  auto o = npt::oracle(
      "a=np.take_along_axis(np.arange(12.).reshape(3,4),"
      "np.array([[3,0,1,2],[0,3,1,2],[2,1,0,3]]),axis=1)");
  if (o) CHECK(allclose(take_along_axis(m, idx, 1), *o, 1e-9, 1e-12, true));
}

// ============================= put =============================
TEST_CASE("put vs numpy") {
  ndarray a = arange(0., 5., 1., kFloat64);
  ndarray idx = ints({0, 2, 4}, {3});
  ndarray vals = arange(10., 40., 10., kFloat64);
  put(a, idx, vals);
  auto o = npt::oracle("a=np.arange(5.); np.put(a,[0,2,4],[10.,20.,30.])");
  if (o) CHECK(allclose(a, *o, 1e-9, 1e-12, true));
}

// ============================= diagonal =============================
TEST_CASE("diagonal main vs numpy") {
  ndarray m = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.diagonal(np.arange(12.).reshape(3,4))");
  if (o) CHECK(allclose(diagonal(m), *o, 1e-9, 1e-12, true));
}
TEST_CASE("diagonal offset vs numpy") {
  ndarray m = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.diagonal(np.arange(12.).reshape(3,4),offset=1)");
  if (o) CHECK(allclose(diagonal(m, 1), *o, 1e-9, 1e-12, true));
}

// ============================= argwhere =============================
TEST_CASE("argwhere vs numpy") {
  ndarray m = arange(0., 9., 1., kFloat64).reshape({3, 3});
  ndarray cond = greater(m, full(m.shape(), 3.0, kFloat64));
  auto o = npt::oracle("a=np.argwhere(np.arange(9.).reshape(3,3)>3)");
  if (o) CHECK(allclose(argwhere(cond), *o, 1e-9, 1e-12, true));
}

// ============================= compress / extract =============================
TEST_CASE("compress axis0 vs numpy") {
  ndarray m = arange(0., 12., 1., kFloat64).reshape({4, 3});
  ndarray cond = ints({1, 0, 1, 1}, {4}).astype(kBool);
  auto o = npt::oracle("a=np.compress([True,False,True,True],np.arange(12.).reshape(4,3),axis=0)");
  if (o) CHECK(allclose(compress(cond, m, 0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("extract vs numpy") {
  ndarray m = arange(0., 9., 1., kFloat64).reshape({3, 3});
  ndarray cond = greater(m, full(m.shape(), 4.0, kFloat64));
  auto o = npt::oracle("m=np.arange(9.).reshape(3,3); a=np.extract(m>4,m)");
  if (o) CHECK(allclose(extract(cond, m), *o, 1e-9, 1e-12, true));
}

// ============================= choose / select =============================
TEST_CASE("choose vs numpy") {
  ndarray idx = ints({0, 1, 2, 0}, {4});
  ndarray c0 = arange(0., 4., 1., kFloat64);
  ndarray c1 = arange(10., 14., 1., kFloat64);
  ndarray c2 = arange(20., 24., 1., kFloat64);
  auto o = npt::oracle("a=np.choose([0,1,2,0],[np.arange(4.),np.arange(10,14.),np.arange(20,24.)])");
  if (o) CHECK(allclose(choose(idx, {c0, c1, c2}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("select vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  ndarray lt2 = less(x, full(x.shape(), 2.0, kFloat64));
  ndarray ge4 = greater_equal(x, full(x.shape(), 4.0, kFloat64));
  auto o = npt::oracle(
      "x=np.arange(6.); a=np.select([x<2,x>=4],[x*10,x*100],default=-1.)");
  if (o) {
    ndarray c0 = multiply(x, full(x.shape(), 10.0, kFloat64));
    ndarray c1 = multiply(x, full(x.shape(), 100.0, kFloat64));
    CHECK(allclose(select({lt2, ge4}, {c0, c1}, -1.0), *o, 1e-9, 1e-12, true));
  }
}

// ============================= ravel / unravel =============================
TEST_CASE("ravel_multi_index vs numpy") {
  ndarray r = ints({0, 1, 2}, {3});
  ndarray c = ints({1, 3, 0}, {3});
  auto o = npt::oracle("a=np.ravel_multi_index([[0,1,2],[1,3,0]],(3,4))");
  if (o) CHECK(allclose(ravel_multi_index({r, c}, {3, 4}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("unravel_index row vs numpy") {
  ndarray flat = ints({1, 5, 11}, {3});
  auto o = npt::oracle("a=np.unravel_index([1,5,11],(3,4))[0]");
  if (o) CHECK(allclose(unravel_index(flat, {3, 4})[0], *o, 1e-9, 1e-12, true));
}
TEST_CASE("unravel_index col vs numpy") {
  ndarray flat = ints({1, 5, 11}, {3});
  auto o = npt::oracle("a=np.unravel_index([1,5,11],(3,4))[1]");
  if (o) CHECK(allclose(unravel_index(flat, {3, 4})[1], *o, 1e-9, 1e-12, true));
}
