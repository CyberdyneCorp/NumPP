// Cases mined from numpy's numpy/lib/tests/test_nanfunctions.py: NaN-aware reductions
// on all-NaN slices, partial-NaN data, and per-axis mixes, vs the live oracle.
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

TEST_CASE("mined nan: all-NaN slice reductions vs numpy") {
  ndarray x = dval({NAN_, NAN_, NAN_});
  const char* PYX = "x=np.array([np.nan,np.nan,np.nan]); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYX) + "a=np.array(" + e + ")");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));  // equal_nan
  };
  chk(nansum(x), "np.nansum(x)");      // 0.0
  chk(nanmean(x), "np.nanmean(x)");    // nan
  chk(nanmax(x), "np.nanmax(x)");      // nan
  chk(nanmin(x), "np.nanmin(x)");      // nan
  chk(nanstd(x), "np.nanstd(x)");      // nan
  chk(nanvar(x), "np.nanvar(x)");      // nan
  chk(nanmedian(x), "np.nanmedian(x)");// nan
}

TEST_CASE("mined nan: partial-NaN reductions vs numpy") {
  ndarray x = dval({1, NAN_, 3, NAN_, 5});
  const char* PYX = "x=np.array([1,np.nan,3,np.nan,5.]); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYX) + "a=np.array(" + e + ")");
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  chk(nansum(x), "np.nansum(x)");      // 9
  chk(nanmean(x), "np.nanmean(x)");    // 3
  chk(nanmax(x), "np.nanmax(x)");      // 5
  chk(nanmin(x), "np.nanmin(x)");      // 1
  chk(nanstd(x), "np.nanstd(x)");
  chk(nanvar(x), "np.nanvar(x)");
  chk(nanmedian(x), "np.nanmedian(x)");// 3
}

TEST_CASE("mined nan: per-axis with an all-NaN row vs numpy") {
  // row 1 is all-NaN -> its nanmean/nanmax/nanmin are nan; nansum is 0
  ndarray A = dval({1, NAN_, 3, NAN_, NAN_, NAN_, 4, 5, 6}).reshape({3, 3});
  const char* PYA = "A=np.array([[1,np.nan,3],[np.nan,np.nan,np.nan],[4,5,6.]]); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYA) + "a=" + e);
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  chk(nansum(A, int64_t{1}), "np.nansum(A,axis=1)");    // [4, 0, 15]
  chk(nanmean(A, int64_t{1}), "np.nanmean(A,axis=1)");  // [2, nan, 5]
  chk(nanmax(A, int64_t{1}), "np.nanmax(A,axis=1)");    // [3, nan, 6]
  chk(nanmin(A, int64_t{1}), "np.nanmin(A,axis=1)");    // [1, nan, 4]
}

TEST_CASE("mined nan: nancumsum / nancumprod treat NaN as 0 / 1 vs numpy") {
  ndarray x = dval({1, NAN_, 3, NAN_, 2});
  auto os = npt::oracle("a=np.nancumsum([1,np.nan,3,np.nan,2.])");  // [1,1,4,4,6]
  auto op = npt::oracle("a=np.nancumprod([1,np.nan,3,np.nan,2.])"); // [1,1,3,3,6]
  if (os) CHECK(allclose(nancumsum(x), *os, 1e-9, 1e-11, true));
  if (op) CHECK(allclose(nancumprod(x), *op, 1e-9, 1e-11, true));
}

TEST_CASE("mined nan: nanargmin / nanargmax ignore NaN vs numpy") {
  ndarray x = dval({NAN_, 3, 1, NAN_, 5, 2});
  auto omin = npt::oracle("a=np.array(np.nanargmin([np.nan,3,1,np.nan,5,2.]))");  // index 2 (val 1)
  auto omax = npt::oracle("a=np.array(np.nanargmax([np.nan,3,1,np.nan,5,2.]))");  // index 4 (val 5)
  if (omin) CHECK(nanargmin(x).item<int64_t>({}) == omin->astype(kInt64).item<int64_t>({}));
  if (omax) CHECK(nanargmax(x).item<int64_t>({}) == omax->astype(kInt64).item<int64_t>({}));
}

TEST_CASE("mined nan: nanpercentile / nanquantile vs numpy") {
  ndarray x = dval({1, NAN_, 3, 7, NAN_, 9, 5});
  auto op = npt::oracle("a=np.array(np.nanpercentile([1,np.nan,3,7,np.nan,9,5.],50))");
  auto oq = npt::oracle("a=np.array(np.nanquantile([1,np.nan,3,7,np.nan,9,5.],0.25))");
  if (op) CHECK(allclose(nanpercentile(x, 50.0), *op, 1e-9, 1e-11, true));
  if (oq) CHECK(allclose(nanquantile(x, 0.25), *oq, 1e-9, 1e-11, true));
}
