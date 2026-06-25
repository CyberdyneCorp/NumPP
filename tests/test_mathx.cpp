#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <vector>

using namespace numpp;

namespace {
ndarray ivals(const std::vector<int64_t>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int64_t>()[i] = v[i];
  return a;
}
ndarray dvals(const std::vector<double>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

// ---- rounding / angle ----
TEST_CASE("around decimals2 vs numpy") {
  ndarray x = dvals({1.23456, 2.71828, -3.14159, 0.005});
  auto o = npt::oracle("a=np.around([1.23456,2.71828,-3.14159,0.005],2)");
  if (o) CHECK(allclose(around(x, 2), *o, 1e-9, 1e-12, true));
}
TEST_CASE("around half-even vs numpy") {
  ndarray x = dvals({0.5, 1.5, 2.5, 3.5, -0.5, -1.5});
  auto o = npt::oracle("a=np.around([0.5,1.5,2.5,3.5,-0.5,-1.5])");
  if (o) CHECK(allclose(around(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("degrees vs numpy") {
  ndarray x = dvals({0.0, M_PI / 2, M_PI, 2 * M_PI});
  auto o = npt::oracle("a=np.degrees([0,np.pi/2,np.pi,2*np.pi])");
  if (o) CHECK(allclose(degrees(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("radians vs numpy") {
  ndarray x = dvals({0.0, 90.0, 180.0, 360.0});
  auto o = npt::oracle("a=np.radians([0,90,180,360])");
  if (o) CHECK(allclose(radians(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("sinc vs numpy") {
  ndarray x = dvals({-1.0, -0.5, 0.0, 0.5, 1.0, 2.0});
  auto o = npt::oracle("a=np.sinc([-1,-0.5,0,0.5,1,2])");
  if (o) CHECK(allclose(sinc(x), *o, 1e-9, 1e-12, true));
}

// ---- integer ----
TEST_CASE("gcd vs numpy") {
  ndarray a = ivals({12, 24, 7, 100});
  ndarray b = ivals({8, 36, 13, 75});
  auto o = npt::oracle("a=np.gcd([12,24,7,100],[8,36,13,75])");
  if (o) CHECK(allclose(gcd(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("lcm vs numpy") {
  ndarray a = ivals({4, 6, 21, 9});
  ndarray b = ivals({6, 8, 6, 12});
  auto o = npt::oracle("a=np.lcm([4,6,21,9],[6,8,6,12])");
  if (o) CHECK(allclose(lcm(a, b), *o, 1e-9, 1e-12, true));
}

// ---- nan / inf ----
TEST_CASE("nan_to_num default vs numpy") {
  ndarray x = dvals({1.0, std::nan(""), std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity()});
  auto o = npt::oracle("a=np.nan_to_num([1.,np.nan,np.inf,-np.inf])");
  if (o) CHECK(allclose(nan_to_num(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("nan_to_num custom vs numpy") {
  ndarray x = dvals({1.0, std::nan(""), std::numeric_limits<double>::infinity()});
  auto o = npt::oracle("a=np.nan_to_num([1.,np.nan,np.inf],nan=-1.0,posinf=1000.0)");
  if (o) CHECK(allclose(nan_to_num(x, -1.0, 1000.0), *o, 1e-9, 1e-12, true));
}

// ---- log-sum-exp / powers / mod ----
TEST_CASE("logaddexp vs numpy") {
  ndarray a = dvals({-1.0, 0.0, 2.0, 10.0});
  ndarray b = dvals({1.0, 3.0, -2.0, 10.0});
  auto o = npt::oracle("a=np.logaddexp([-1,0,2,10],[1,3,-2,10])");
  if (o) CHECK(allclose(logaddexp(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("logaddexp2 vs numpy") {
  ndarray a = dvals({-1.0, 0.0, 2.0});
  ndarray b = dvals({1.0, 3.0, -2.0});
  auto o = npt::oracle("a=np.logaddexp2([-1,0,2],[1,3,-2])");
  if (o) CHECK(allclose(logaddexp2(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("float_power vs numpy") {
  ndarray a = dvals({2.0, 3.0, 4.0});
  ndarray b = dvals({0.5, 2.0, -1.0});
  auto o = npt::oracle("a=np.float_power([2,3,4],[0.5,2,-1])");
  if (o) CHECK(allclose(float_power(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("fmod vs numpy") {
  ndarray a = dvals({5.3, -5.3, 7.0});
  ndarray b = dvals({2.0, 2.0, -3.0});
  auto o = npt::oracle("a=np.fmod([5.3,-5.3,7.0],[2.0,2.0,-3.0])");
  if (o) CHECK(allclose(fmod(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("heaviside vs numpy") {
  ndarray x = dvals({-1.5, 0.0, 2.0});
  ndarray h = full({3}, 0.5, kFloat64);
  auto o = npt::oracle("a=np.heaviside([-1.5,0.,2.],0.5)");
  if (o) CHECK(allclose(heaviside(x, h), *o, 1e-9, 1e-12, true));
}

// ---- decompositions ----
TEST_CASE("modf frac vs numpy") {
  ndarray x = dvals({1.5, -2.25, 3.0});
  auto o = npt::oracle("a=np.modf([1.5,-2.25,3.0])[0]");
  if (o) CHECK(allclose(modf(x).first, *o, 1e-9, 1e-12, true));
}
TEST_CASE("modf int vs numpy") {
  ndarray x = dvals({1.5, -2.25, 3.0});
  auto o = npt::oracle("a=np.modf([1.5,-2.25,3.0])[1]");
  if (o) CHECK(allclose(modf(x).second, *o, 1e-9, 1e-12, true));
}
TEST_CASE("frexp mantissa vs numpy") {
  ndarray x = dvals({1.0, 8.0, 0.25, -3.0});
  auto o = npt::oracle("a=np.frexp([1.,8.,0.25,-3.])[0]");
  if (o) CHECK(allclose(frexp(x).first, *o, 1e-9, 1e-12, true));
}
TEST_CASE("frexp exponent vs numpy") {
  ndarray x = dvals({1.0, 8.0, 0.25, -3.0});
  auto o = npt::oracle("a=np.frexp([1.,8.,0.25,-3.])[1]");
  if (o) CHECK(allclose(frexp(x).second, *o, 1e-9, 1e-12, true));
}
TEST_CASE("ldexp vs numpy") {
  ndarray m = dvals({0.5, 1.0, 0.75});
  ndarray e = dvals({3.0, 4.0, 2.0});
  auto o = npt::oracle("a=np.ldexp([0.5,1.0,0.75],[3,4,2])");
  if (o) CHECK(allclose(ldexp(m, e), *o, 1e-9, 1e-12, true));
}
TEST_CASE("divmod quotient vs numpy") {
  ndarray a = dvals({7.0, -7.0, 13.0});
  ndarray b = dvals({3.0, 3.0, -4.0});
  auto o = npt::oracle("a=np.divmod([7.,-7.,13.],[3.,3.,-4.])[0]");
  if (o) CHECK(allclose(divmod(a, b).first, *o, 1e-9, 1e-12, true));
}
TEST_CASE("divmod remainder vs numpy") {
  ndarray a = dvals({7.0, -7.0, 13.0});
  ndarray b = dvals({3.0, 3.0, -4.0});
  auto o = npt::oracle("a=np.divmod([7.,-7.,13.],[3.,3.,-4.])[1]");
  if (o) CHECK(allclose(divmod(a, b).second, *o, 1e-9, 1e-12, true));
}

// ---- misc ----
TEST_CASE("unwrap vs numpy") {
  ndarray x = dvals({0.0, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0});
  auto o = npt::oracle("a=np.unwrap([0.,1.,2.,3.,-3.,-2.,-1.])");
  if (o) CHECK(allclose(unwrap(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("i0 vs numpy") {
  ndarray x = dvals({0.0, 0.5, 1.0, 2.0, 3.0});
  auto o = npt::oracle("a=np.i0([0.,0.5,1.,2.,3.])");
  if (o) CHECK(allclose(i0(x), *o, 1e-7, 1e-9, true));
}
TEST_CASE("nextafter vs numpy") {
  ndarray a = dvals({1.0, 2.0, -1.0});
  ndarray b = dvals({2.0, 1.0, 0.0});
  auto o = npt::oracle("a=np.nextafter([1.,2.,-1.],[2.,1.,0.])");
  if (o) CHECK(allclose(nextafter(a, b), *o, 0.0, 0.0, true));
}
TEST_CASE("spacing vs numpy") {
  ndarray x = dvals({1.0, 1000.0, -1.0});
  auto o = npt::oracle("a=np.spacing([1.,1000.,-1.])");
  if (o) CHECK(allclose(spacing(x), *o, 1e-6, 0.0, true));
}
