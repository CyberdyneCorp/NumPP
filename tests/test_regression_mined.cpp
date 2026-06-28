// Cases mined from numpy's numpy/_core/tests/test_regression.py and assorted
// historical gotchas: NaN ordering in sort/argsort/argmax, round half-to-even
// (incl. decimals and negative decimals), sinc/heaviside, multi-axis roll,
// diff n=0, empty-reduction identities, and the int**negative ValueError.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace numpp;

namespace {
const double NAN_ = std::numeric_limits<double>::quiet_NaN();
ndarray V(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined regression: NaN sorts to the end (sort/argsort/argmax) vs numpy") {
  auto os = npt::oracle("a=np.sort([3,np.nan,1,np.nan,2.])");
  if (os) CHECK(allclose(sort(V({3, NAN_, 1, NAN_, 2})), *os, 0, 0, true));
  auto oa = npt::oracle("a=np.argsort([3,np.nan,1,2.])");
  if (oa) CHECK(allclose(argsort(V({3, NAN_, 1, 2})).astype(kFloat64), *oa, 0, 0, true));
  auto om = npt::oracle("a=np.array(np.argmax([1,np.nan,3.]))");
  if (om) CHECK(allclose(argmax(V({1, NAN_, 3}), std::nullopt).astype(kFloat64), *om, 0, 0, true));
}

TEST_CASE("mined regression: round half-to-even (decimals and negative) vs numpy") {
  auto o0 = npt::oracle("a=np.round([0.5,1.5,2.5,3.5,-0.5,-1.5])");
  if (o0) CHECK(allclose(around(V({0.5, 1.5, 2.5, 3.5, -0.5, -1.5})), *o0, 0, 0, true));
  auto o2 = npt::oracle("a=np.round([2.675,2.665,1.005,0.125],2)");
  if (o2) CHECK(allclose(around(V({2.675, 2.665, 1.005, 0.125}), 2), *o2, 1e-12, 1e-12, true));
  auto on = npt::oracle("a=np.round([12345.,67890.],-2)");
  if (on) CHECK(allclose(around(V({12345, 67890}), -2), *on, 0, 0, true));
}

TEST_CASE("mined regression: sinc and heaviside vs numpy") {
  auto os = npt::oracle("a=np.sinc([0,0.5,1,1.5])");
  if (os) CHECK(allclose(sinc(V({0, 0.5, 1, 1.5})), *os, 1e-12, 1e-12, true));
  auto oh = npt::oracle("a=np.heaviside([-1.,0.,2.],0.5)");
  if (oh) CHECK(allclose(heaviside(V({-1, 0, 2}), full(Shape{3}, 0.5, kFloat64)), *oh, 0, 0, true));
}

TEST_CASE("mined regression: multi-axis roll and diff n=0 vs numpy") {
  ndarray m = arange(0.0, 6.0, 1.0).reshape({2, 3});
  auto orr = npt::oracle("a=np.roll(np.arange(6.).reshape(2,3),1,axis=1)");
  if (orr) CHECK(allclose(roll(m, 1, 1), *orr, 0, 0, true));
  auto od = npt::oracle("a=np.diff([1,2,4.],n=0)");
  if (od) CHECK(allclose(diff(V({1, 2, 4}), 0, -1), *od, 0, 0, true));
}

TEST_CASE("mined regression: empty/degenerate reductions vs numpy") {
  // mean of an empty array is nan (with a runtime warning in numpy).
  CHECK(std::isnan(mean(V({}), std::nullopt).item<double>({})));
  // std of a single element is 0.
  CHECK(numpp::std(V({5}), std::nullopt).item<double>({}) == 0.0);
  // clip with reversed bounds collapses to the lower bound (numpy applies max then min).
  ndarray a = arange(0.0, 6.0, 1.0);
  ndarray r = clip(a, full(a.shape(), 5.0, kFloat64), full(a.shape(), 1.0, kFloat64));
  auto o = npt::oracle("a=np.clip(np.arange(6.),5,1)");
  if (o) CHECK(allclose(r, *o, 0, 0, true));
}

TEST_CASE("mined regression: integers to negative integer powers raise (numpy parity)") {
  ndarray b(Shape{2}, kInt64, Order::C);
  b.typed_data<int64_t>()[0] = 2;
  b.typed_data<int64_t>()[1] = 3;
  ndarray e(Shape{2}, kInt64, Order::C);
  e.typed_data<int64_t>()[0] = -1;
  e.typed_data<int64_t>()[1] = -1;
  CHECK_THROWS_AS(power(b, e), value_error);
}

TEST_CASE("mined regression: partition places the kth element correctly (property)") {
  // numpy does not specify the order of the non-kth elements, so we verify the
  // partition contract rather than exact equality.
  ndarray p = partition(V({3, 1, 4, 1, 5, 9, 2}), 3, std::nullopt);
  const double* d = p.typed_data<double>();
  const double kth = d[3];
  CHECK(kth == 3.0);  // 4th smallest of {1,1,2,3,4,5,9}
  for (int64_t i = 0; i < 3; ++i) CHECK(d[i] <= kth);
  for (int64_t i = 4; i < 7; ++i) CHECK(d[i] >= kth);
}
