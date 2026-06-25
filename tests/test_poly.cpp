#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <vector>

using namespace numpp;

namespace {
ndarray dvals(const std::vector<double>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

// ---- signal ----
TEST_CASE("convolve full vs numpy") {
  ndarray a = dvals({1, 2, 3}), v = dvals({0, 1, 0.5});
  auto o = npt::oracle("a=np.convolve([1,2,3],[0,1,0.5],'full')");
  if (o) CHECK(allclose(convolve(a, v, "full"), *o, 1e-9, 1e-12, true));
}
TEST_CASE("convolve same vs numpy") {
  ndarray a = dvals({1, 2, 3, 4}), v = dvals({1, 1, 1});
  auto o = npt::oracle("a=np.convolve([1,2,3,4],[1,1,1],'same')");
  if (o) CHECK(allclose(convolve(a, v, "same"), *o, 1e-9, 1e-12, true));
}
TEST_CASE("convolve valid vs numpy") {
  ndarray a = dvals({1, 2, 3, 4, 5}), v = dvals({1, 2});
  auto o = npt::oracle("a=np.convolve([1,2,3,4,5],[1,2],'valid')");
  if (o) CHECK(allclose(convolve(a, v, "valid"), *o, 1e-9, 1e-12, true));
}
TEST_CASE("correlate full vs numpy") {
  ndarray a = dvals({1, 2, 3}), v = dvals({0, 1, 0.5});
  auto o = npt::oracle("a=np.correlate([1,2,3],[0,1,0.5],'full')");
  if (o) CHECK(allclose(correlate(a, v, "full"), *o, 1e-9, 1e-12, true));
}
TEST_CASE("correlate valid vs numpy") {
  ndarray a = dvals({1, 2, 3, 4, 5}), v = dvals({1, 2});
  auto o = npt::oracle("a=np.correlate([1,2,3,4,5],[1,2],'valid')");
  if (o) CHECK(allclose(correlate(a, v, "valid"), *o, 1e-9, 1e-12, true));
}
TEST_CASE("interp vs numpy") {
  ndarray x = dvals({0.0, 0.5, 1.5, 2.5, 3.0, 4.0});
  ndarray xp = dvals({0.0, 1.0, 2.0, 3.0});
  ndarray fp = dvals({0.0, 10.0, 20.0, 30.0});
  auto o = npt::oracle("a=np.interp([0,0.5,1.5,2.5,3,4],[0,1,2,3],[0,10,20,30.])");
  if (o) CHECK(allclose(interp(x, xp, fp), *o, 1e-9, 1e-12, true));
}

// ---- polynomials ----
TEST_CASE("polyval vs numpy") {
  ndarray p = dvals({1, -2, 0, 3});  // x^3 - 2x^2 + 3
  ndarray x = dvals({-1, 0, 1, 2, 3});
  auto o = npt::oracle("a=np.polyval([1,-2,0,3],[-1,0,1,2,3])");
  if (o) CHECK(allclose(polyval(p, x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polyadd vs numpy") {
  ndarray a = dvals({1, 2}), b = dvals({-1, 3, 4});
  auto o = npt::oracle("a=np.polyadd([1,2],[-1,3,4])");
  if (o) CHECK(allclose(polyadd(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polysub vs numpy") {
  ndarray a = dvals({1, 2, 3}), b = dvals({1, 2});
  auto o = npt::oracle("a=np.polysub([1,2,3],[1,2])");
  if (o) CHECK(allclose(polysub(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polymul vs numpy") {
  ndarray a = dvals({1, 2, 3}), b = dvals({1, 1});
  auto o = npt::oracle("a=np.polymul([1,2,3],[1,1])");
  if (o) CHECK(allclose(polymul(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polydiv quotient vs numpy") {
  ndarray a = dvals({1, 0, -1, 2}), b = dvals({1, 1});
  auto o = npt::oracle("a=np.polydiv([1,0,-1,2],[1,1])[0]");
  if (o) CHECK(allclose(polydiv(a, b).first, *o, 1e-9, 1e-12, true));
}
TEST_CASE("polydiv remainder vs numpy") {
  ndarray a = dvals({1, 0, -1, 2}), b = dvals({1, 1});
  auto o = npt::oracle("a=np.polydiv([1,0,-1,2],[1,1])[1]");
  if (o) CHECK(allclose(polydiv(a, b).second, *o, 1e-9, 1e-12, true));
}
TEST_CASE("polyder vs numpy") {
  ndarray p = dvals({1, -2, 0, 3});
  auto o = npt::oracle("a=np.polyder([1,-2,0,3])");
  if (o) CHECK(allclose(polyder(p), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polyder order2 vs numpy") {
  ndarray p = dvals({1, -2, 0, 3});
  auto o = npt::oracle("a=np.polyder([1,-2,0,3],2)");
  if (o) CHECK(allclose(polyder(p, 2), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polyint vs numpy") {
  ndarray p = dvals({3, -2, 1});
  auto o = npt::oracle("a=np.polyint([3,-2,1])");
  if (o) CHECK(allclose(polyint(p), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polyint with k vs numpy") {
  ndarray p = dvals({3, -2, 1});
  auto o = npt::oracle("a=np.polyint([3,-2,1],k=5)");
  if (o) CHECK(allclose(polyint(p, 1, 5.0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly from roots vs numpy") {
  ndarray r = dvals({1, 2, 3});
  auto o = npt::oracle("a=np.poly([1,2,3])");
  if (o) CHECK(allclose(poly(r), *o, 1e-9, 1e-12, true));
}
TEST_CASE("roots sorted vs numpy") {
  ndarray p = dvals({1, -6, 11, -6});  // roots 1,2,3
  auto o = npt::oracle("a=np.sort(np.roots([1,-6,11,-6]).real)");
  if (o) CHECK(allclose(sort(roots(p)), *o, 1e-6, 1e-9, true));
}
TEST_CASE("polyfit recovers coeffs vs numpy") {
  ndarray x = arange(0., 8., 1., kFloat64);
  ndarray p = dvals({2, -3, 1});  // 2x^2 - 3x + 1
  ndarray y = polyval(p, x);
  auto o = npt::oracle(
      "x=np.arange(8.); y=np.polyval([2,-3,1],x); a=np.polyfit(x,y,2)");
  if (o) CHECK(allclose(polyfit(x, y, 2), *o, 1e-6, 1e-8, true));
}
