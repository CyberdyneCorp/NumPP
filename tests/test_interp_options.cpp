#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <vector>

using namespace numpp;

namespace {
ndarray dvals(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("interp default boundaries still match numpy") {
  ndarray x = dvals({0, 1, 1.5, 3, 4});
  ndarray xp = dvals({1, 2, 3}), fp = dvals({10, 20, 30});
  auto o = npt::oracle("a=np.interp([0,1,1.5,3,4],[1,2,3],[10,20,30.])");
  if (o) CHECK(allclose(interp(x, xp, fp), *o, 1e-12, 1e-12, true));
}

TEST_CASE("interp left/right fill values match numpy") {
  ndarray x = dvals({0, 1, 1.5, 3, 4});
  ndarray xp = dvals({1, 2, 3}), fp = dvals({10, 20, 30});
  ndarray got = interp(x, xp, fp, -1.0, -2.0);
  auto o = npt::oracle("a=np.interp([0,1,1.5,3,4],[1,2,3],[10,20,30.],left=-1,right=-2)");
  if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
}

TEST_CASE("interp returns the endpoint (not left) exactly at xp[0]") {
  ndarray x = dvals({1.0});
  ndarray xp = dvals({1, 2, 3}), fp = dvals({10, 20, 30});
  ndarray got = interp(x, xp, fp, -1.0, -2.0);  // left=-1 must NOT apply at x==xp[0]
  CHECK(got.item<double>({0}) == 10.0);
}

TEST_CASE("interp periodic mode matches numpy") {
  ndarray x = dvals({-180, -90, 0, 90, 270, 360, 450});
  ndarray xp = dvals({0, 90, 180, 270}), fp = dvals({1, 2, 3, 4});
  ndarray got = interp(x, xp, fp, std::nullopt, std::nullopt, 360.0);
  auto o = npt::oracle(
      "a=np.interp([-180,-90,0,90,270,360,450],[0,90,180,270],[1,2,3,4.],period=360)");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}

TEST_CASE("interp periodic with unsorted xp matches numpy") {
  ndarray x = dvals({10, 100, 200, 350});
  ndarray xp = dvals({270, 0, 180, 90}), fp = dvals({4, 1, 3, 2});  // unsorted
  ndarray got = interp(x, xp, fp, std::nullopt, std::nullopt, 360.0);
  auto o = npt::oracle(
      "a=np.interp([10,100,200,350],[270,0,180,90],[4,1,3,2.],period=360)");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}
