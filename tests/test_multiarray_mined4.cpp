// Deeper test_multiarray.py mining: clip edge cases (reversed bounds, one-sided),
// ravel/flatten C vs F order, where broadcasting, argwhere/flatnonzero.
#include "numpp/numpp.hpp"
#include "numpp/indexing/indexing.hpp"
#include "numpp/sorting/sorting.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <limits>
#include <vector>

using namespace numpp;

namespace {
const double INF = std::numeric_limits<double>::infinity();
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray bvec(const std::vector<bool>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kBool, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<bool>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined multiarray: clip with reversed bounds (lo>hi) vs numpy") {
  ndarray a = dval({0, 3, 10, -2, 7});
  ndarray lo = dval({5, 5, 5, 5, 5});
  ndarray hi = dval({1, 1, 1, 1, 1});
  // numpy clip = minimum(maximum(a, lo), hi); when lo>hi the hi bound wins.
  auto o = npt::oracle("a=np.clip(np.array([0,3,10,-2,7.]),5,1)");
  if (o) CHECK(allclose(clip(a, lo, hi), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined multiarray: one-sided clip vs numpy") {
  ndarray a = dval({-3, 0, 5, 12});
  ndarray loonly = dval({0, 0, 0, 0});
  ndarray hiinf = dval({INF, INF, INF, INF});
  ndarray loneg = dval({-INF, -INF, -INF, -INF});
  ndarray hionly = dval({5, 5, 5, 5});
  auto o1 = npt::oracle("a=np.clip(np.array([-3,0,5,12.]),0,None)");
  auto o2 = npt::oracle("a=np.clip(np.array([-3,0,5,12.]),None,5)");
  if (o1) CHECK(allclose(clip(a, loonly, hiinf), *o1, 1e-12, 1e-12, true));
  if (o2) CHECK(allclose(clip(a, loneg, hionly), *o2, 1e-12, 1e-12, true));
}

TEST_CASE("mined multiarray: ravel C vs F order vs numpy") {
  ndarray A = arange(0.0, 6.0, 1.0).reshape({2, 3});
  auto oc = npt::oracle("A=np.arange(6.).reshape(2,3); a=A.ravel(order='C')");
  auto of = npt::oracle("A=np.arange(6.).reshape(2,3); a=A.ravel(order='F')");
  if (oc) CHECK(allclose(A.ravel(Order::C), *oc, 0, 0, true));
  if (of) CHECK(allclose(A.ravel(Order::F), *of, 0, 0, true));
}

TEST_CASE("mined multiarray: flatten F order on a transposed view vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4}).transpose();  // (4,3) non-contiguous
  auto oc = npt::oracle("A=np.arange(12.).reshape(3,4).T; a=A.flatten(order='C')");
  auto of = npt::oracle("A=np.arange(12.).reshape(3,4).T; a=A.flatten(order='F')");
  if (oc) CHECK(allclose(A.flatten(Order::C), *oc, 0, 0, true));
  if (of) CHECK(allclose(A.flatten(Order::F), *of, 0, 0, true));
}

TEST_CASE("mined multiarray: where with broadcasting vs numpy") {
  ndarray cond = bvec({true, false, true, false, true, false}).reshape({2, 3});
  ndarray x = dval({10, 20, 30}).reshape({1, 3});   // broadcasts over rows
  ndarray y = dval({-1, -2}).reshape({2, 1});       // broadcasts over cols
  ndarray got = where(cond, x, y);
  CHECK((got.shape() == Shape{2, 3}));
  auto o = npt::oracle(
      "c=np.array([[True,False,True],[False,True,False]]); "
      "x=np.array([[10,20,30.]]); y=np.array([[-1.],[-2.]]); a=np.where(c,x,y)");
  if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined multiarray: argwhere / flatnonzero vs numpy") {
  ndarray A = dval({0, 3, 0, 0, 5, 7, 0, 2, 0}).reshape({3, 3});
  auto oa = npt::oracle("A=np.array([[0,3,0],[0,5,7],[0,2,0.]]); a=np.argwhere(A)");
  auto of = npt::oracle("A=np.array([[0,3,0],[0,5,7],[0,2,0.]]); a=np.flatnonzero(A)");
  if (oa) CHECK(allclose(argwhere(A).astype(kFloat64), oa->astype(kFloat64), 0, 0, true));
  if (of) CHECK(allclose(flatnonzero(A).astype(kFloat64), of->astype(kFloat64), 0, 0, true));
}
