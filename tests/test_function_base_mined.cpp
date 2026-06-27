// Cases mined from numpy's numpy/lib/tests/test_function_base.py: interp (basic,
// left/right fills, periodic), trapz (dx and explicit x), bincount (counts,
// weights, minlength), percentile/median, gradient, unwrap — vs the live oracle.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <string>
#include <vector>

using namespace numpp;

namespace {
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray i64v(const std::vector<int64_t>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int64_t>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined function_base: interp basic / left-right / periodic vs numpy") {
  ndarray xp = dval({0, 1, 2, 3}), fp = dval({0, 10, 20, 30});
  ndarray x = dval({0.5, 1.5, 2.5});
  auto o1 = npt::oracle("a=np.interp([0.5,1.5,2.5],[0,1,2,3.],[0,10,20,30.])");
  if (o1) CHECK(allclose(interp(x, xp, fp), *o1, 1e-12, 1e-12, true));

  ndarray x2 = dval({-1, 5});
  ndarray xp2 = dval({0, 1, 2}), fp2 = dval({10, 20, 30});
  auto o2 = npt::oracle("a=np.interp([-1,5.],[0,1,2.],[10,20,30.],left=-99,right=99)");
  if (o2) CHECK(allclose(interp(x2, xp2, fp2, -99.0, 99.0), *o2, 1e-12, 1e-12, true));

  ndarray x3 = dval({1.5, 4.0});
  auto o3 = npt::oracle("a=np.interp([1.5,4.0],[0,1,2,3.],[0,10,20,30.],period=4)");
  if (o3) CHECK(allclose(interp(x3, xp, fp, std::nullopt, std::nullopt, 4.0), *o3, 1e-12, 1e-12, true));
}

TEST_CASE("mined function_base: trapz with dx and explicit x vs numpy") {
  ndarray y = dval({1, 2, 3, 4});
  auto o1 = npt::oracle("import warnings; warnings.simplefilter('ignore'); a=np.array(np.trapz([1,2,3,4.]))");
  if (o1) CHECK(allclose(trapz(y), *o1, 1e-12, 1e-12, true));

  ndarray y2 = dval({1, 2, 3}), x2 = dval({0, 2, 4});
  auto o2 = npt::oracle("import warnings; warnings.simplefilter('ignore'); a=np.array(np.trapz([1,2,3.],x=[0,2,4.]))");
  if (o2) CHECK(allclose(trapz(y2, x2), *o2, 1e-12, 1e-12, true));
}

TEST_CASE("mined function_base: bincount counts / weights / minlength vs numpy") {
  ndarray x = i64v({0, 1, 1, 2, 2, 2});
  auto o1 = npt::oracle("a=np.bincount([0,1,1,2,2,2])");
  if (o1) CHECK(allclose(bincount(x).astype(kFloat64), *o1, 0, 0, true));

  ndarray xw = i64v({0, 1, 1, 2});
  ndarray w = dval({0.5, 1, 1, 2});
  auto o2 = npt::oracle("a=np.bincount([0,1,1,2],weights=[0.5,1,1,2.])");
  if (o2) CHECK(allclose(bincount(xw, &w), *o2, 1e-12, 1e-12, true));

  ndarray xm = i64v({1, 2});
  auto o3 = npt::oracle("a=np.bincount([1,2],minlength=5)");
  if (o3) CHECK(allclose(bincount(xm, nullptr, 5).astype(kFloat64), *o3, 0, 0, true));
}

TEST_CASE("mined function_base: percentile / median vs numpy") {
  ndarray a = dval({3, 1, 2, 5, 4});
  auto chk = [&](const ndarray& got, const std::string& py) {
    auto o = npt::oracle("a=np.array(" + py + ")");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(percentile(a, 25.0), "np.percentile([3,1,2,5,4.],25)");
  chk(percentile(a, 0.0), "np.percentile([3,1,2,5,4.],0)");
  chk(percentile(a, 100.0), "np.percentile([3,1,2,5,4.],100)");
  ndarray e = dval({1, 2, 3, 4});
  chk(median(e), "np.median([1,2,3,4.])");
  chk(median(a), "np.median([3,1,2,5,4.])");
}

TEST_CASE("mined function_base: gradient (1-D, unit spacing) vs numpy") {
  ndarray y = dval({1, 2, 4, 7, 11});
  auto o = npt::oracle("a=np.gradient([1.,2,4,7,11])");
  if (o) CHECK(allclose(gradient(y), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined function_base: unwrap phase vs numpy") {
  ndarray p = dval({0, 0.1, 6.0, 0.2});
  auto o = npt::oracle("a=np.unwrap([0,0.1,6.0,0.2])");
  if (o) CHECK(allclose(unwrap(p), *o, 1e-12, 1e-12, true));
}
