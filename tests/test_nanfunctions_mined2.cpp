// Deeper test_nanfunctions.py mining: all-NaN median/percentile/quantile, nanstd/
// nanvar with ddof reducing the degrees of freedom, and the nanargmin/nanargmax
// all-NaN-slice ValueError.
#include "numpp/numpp.hpp"
#include "numpp/stats/stats.hpp"
#include "numpp/stats/stats_extra.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <limits>
#include <string>
#include <vector>

using namespace numpp;

namespace {
const double NAN_ = std::numeric_limits<double>::quiet_NaN();
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined nan: all-NaN median/percentile/quantile are nan vs numpy") {
  ndarray x = dval({NAN_, NAN_});
  auto om = npt::oracle("a=np.array(np.nanmedian([np.nan,np.nan]))");
  auto op = npt::oracle("a=np.array(np.nanpercentile([np.nan,np.nan],50))");
  auto oq = npt::oracle("a=np.array(np.nanquantile([np.nan,np.nan],0.25))");
  if (om) CHECK(allclose(nanmedian(x), *om, 0, 0, true));
  if (op) CHECK(allclose(nanpercentile(x, 50.0), *op, 0, 0, true));
  if (oq) CHECK(allclose(nanquantile(x, 0.25), *oq, 0, 0, true));
}

TEST_CASE("mined nan: nanstd/nanvar with ddof >= count are nan vs numpy") {
  ndarray a = dval({1, NAN_});         // 1 non-NaN value
  ndarray b = dval({1, 2, NAN_});      // 2 non-NaN values
  auto os = npt::oracle("a=np.array(np.nanstd(np.array([1.,np.nan]),ddof=1))");
  auto ov = npt::oracle("a=np.array(np.nanvar(np.array([1.,2,np.nan]),ddof=2))");
  if (os) CHECK(allclose(nanstd(a, std::nullopt, false, 1), *os, 0, 0, true));   // nan
  if (ov) CHECK(allclose(nanvar(b, std::nullopt, false, 2), *ov, 0, 0, true));   // nan
}

TEST_CASE("mined nan: nanargmin/nanargmax raise on an all-NaN slice (#94 follow-up)") {
  ndarray x = dval({NAN_, NAN_, NAN_});
  CHECK_THROWS_AS(nanargmax(x), value_error);
  CHECK_THROWS_AS(nanargmin(x), value_error);
}

TEST_CASE("mined nan: nanargmin/nanargmax raise when any row is all-NaN") {
  // row 1 is all-NaN -> numpy raises ValueError over axis=1
  ndarray A = dval({1, NAN_, 3, NAN_, NAN_, NAN_, 4, 5, 6}).reshape({3, 3});
  CHECK_THROWS_AS(nanargmax(A, int64_t{1}), value_error);
  CHECK_THROWS_AS(nanargmin(A, int64_t{1}), value_error);
}

TEST_CASE("mined nan: nanargmin/nanargmax still work when some values are present") {
  ndarray A = dval({1, NAN_, 3, 9, NAN_, 7, 4, 5, NAN_}).reshape({3, 3});
  auto omax = npt::oracle(
      "A=np.array([[1,np.nan,3],[9,np.nan,7],[4,5,np.nan]]); a=np.nanargmax(A,axis=1)");
  auto omin = npt::oracle(
      "A=np.array([[1,np.nan,3],[9,np.nan,7],[4,5,np.nan]]); a=np.nanargmin(A,axis=1)");
  if (omax) CHECK(allclose(nanargmax(A, int64_t{1}).astype(kFloat64), omax->astype(kFloat64), 0, 0, true));
  if (omin) CHECK(allclose(nanargmin(A, int64_t{1}).astype(kFloat64), omin->astype(kFloat64), 0, 0, true));
}
