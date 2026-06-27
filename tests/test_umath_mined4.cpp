// Remaining test_umath.py mining: out=/where= aliasing, mixed-dtype comparison
// broadcasting, INT_MIN/-1 integer overflow, and reduce/accumulate vs the oracle.
#include "numpp/numpp.hpp"
#include "numpp/stats/stats.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cstdint>
#include <limits>
#include <vector>

using namespace numpp;

namespace {
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray i64(const std::vector<int64_t>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int64_t>()[i] = v[i];
  return a;
}
ndarray i8(const std::vector<int8_t>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kInt8, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int8_t>()[i] = v[i];
  return a;
}
ndarray bvec(const std::vector<bool>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kBool, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<bool>()[i] = v[i];
  return a;
}
const int64_t IMIN = std::numeric_limits<int64_t>::min();
}  // namespace

TEST_CASE("mined umath: out= writes into and returns the out array") {
  ndarray a = dval({1, 2, 3}), b = dval({10, 20, 30});
  ndarray out = dval({0, 0, 0});
  ndarray r = add(a, b, out);
  auto o = npt::oracle("a=np.add([1,2,3.],[10,20,30.],out=np.zeros(3))");
  if (o) CHECK(allclose(out, *o, 1e-12, 1e-12, true));
  CHECK(allclose(r, out, 0, 0, true));  // returns the same out
}

TEST_CASE("mined umath: in-place add aliasing out=first-input") {
  ndarray a = dval({1, 2, 3}), b = dval({10, 20, 30});
  add(a, b, a);  // a += b
  auto o = npt::oracle("a=np.array([1,2,3.]); a+=np.array([10,20,30.])");
  if (o) CHECK(allclose(a, *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined umath: where= leaves unselected out elements unchanged vs numpy") {
  ndarray a = dval({1, 2, 3, 4}), b = dval({10, 20, 30, 40});
  ndarray out = dval({100, 100, 100, 100});
  ndarray mask = bvec({true, false, true, false});
  add(a, b, out, &mask);
  auto o = npt::oracle(
      "out=np.full(4,100.); np.add([1,2,3,4.],[10,20,30,40.],out=out,"
      "where=np.array([True,False,True,False])); a=out");
  if (o) CHECK(allclose(out, *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined umath: mixed-dtype comparison vs numpy") {
  ndarray ai = i8({1, 2, 3});
  ndarray bf = dval({1.0, 2.5, 3.0});
  auto o = npt::oracle("a=(np.array([1,2,3],dtype=np.int8)==np.array([1,2.5,3.]))");
  if (o) CHECK(allclose(equal(ai, bf).astype(kFloat64), *o, 0, 0, true));
}

TEST_CASE("mined umath: comparison broadcasting (3,1) vs (1,4) matches numpy") {
  ndarray a = i64({1, 2, 3}).reshape({3, 1});
  ndarray b = i64({1, 2, 3, 4}).reshape({1, 4});
  ndarray got = less(a, b);
  CHECK((got.shape() == Shape{3, 4}));
  auto o = npt::oracle(
      "A=np.arange(1,4).reshape(3,1); B=np.arange(1,5).reshape(1,4); a=(A<B)");
  if (o) CHECK(allclose(got.astype(kFloat64), *o, 0, 0, true));
}

TEST_CASE("mined umath: INT_MIN // -1 and INT_MIN % -1 do not overflow (vs numpy)") {
  ndarray a = i64({IMIN, -10, 10, IMIN});
  ndarray b = i64({-1, -1, -1, 1});
  auto od = npt::oracle(
      "a=np.array([-2**63,-10,10,-2**63],dtype=np.int64)//np.array([-1,-1,-1,1],dtype=np.int64)");
  auto om = npt::oracle(
      "a=np.remainder(np.array([-2**63,-10,10,-2**63],dtype=np.int64),"
      "np.array([-1,-1,-1,1],dtype=np.int64))");
  if (od) CHECK(allclose(floor_divide(a, b).astype(kFloat64), od->astype(kFloat64), 0, 0, true));
  if (om) CHECK(allclose(mod(a, b).astype(kFloat64), om->astype(kFloat64), 0, 0, true));
}

TEST_CASE("mined umath: reduce (sum/prod/amax/amin) over an axis vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4});
  const char* PYA = "A=np.arange(12.).reshape(3,4); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYA) + "a=" + e);
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(sum(A, int64_t{0}), "A.sum(axis=0)");
  chk(prod(A, int64_t{1}), "A.prod(axis=1)");
  chk(amax(A, int64_t{0}), "A.max(axis=0)");
  chk(amin(A, int64_t{1}), "A.min(axis=1)");
}

TEST_CASE("mined umath: empty-array reduce identities vs numpy") {
  ndarray e = dval({});
  auto os = npt::oracle("a=np.array(np.sum(np.array([])))");
  auto op = npt::oracle("a=np.array(np.prod(np.array([])))");
  if (os) CHECK(allclose(sum(e), *os, 0, 0, true));   // 0.0
  if (op) CHECK(allclose(prod(e), *op, 0, 0, true));  // 1.0
}

TEST_CASE("mined umath: accumulate (cumsum/cumprod) over an axis vs numpy") {
  ndarray A = arange(1.0, 7.0, 1.0).reshape({2, 3});
  const char* PYA = "A=np.arange(1,7.).reshape(2,3); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYA) + "a=" + e);
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(cumsum(A, int64_t{1}), "np.cumsum(A,axis=1)");
  chk(cumprod(A, int64_t{0}), "np.cumprod(A,axis=0)");
}
