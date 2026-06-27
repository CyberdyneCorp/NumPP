// Deeper cases mined from numpy's numpy/_core/tests/test_umath.py: integer
// floor_divide/remainder (signs + divide-by-zero), float_power, complex branch
// cuts, bitwise ufuncs, and comparison ufuncs with NaN — all vs the live oracle.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <complex>
#include <limits>
#include <string>
#include <vector>

using namespace numpp;
using cd = std::complex<double>;

namespace {
const double INF = std::numeric_limits<double>::infinity();
const double NAN_ = std::numeric_limits<double>::quiet_NaN();
ndarray ival(const std::vector<int64_t>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int64_t>()[i] = v[i];
  return a;
}
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray cval(const std::vector<cd>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kComplex128, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<cd>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined umath: integer floor_divide signs + divide-by-zero vs numpy") {
  ndarray a = ival({7, -7, 7, -7, 5, -5});
  ndarray b = ival({3, 3, -3, -3, 0, 0});
  auto o = npt::oracle(
      "a=np.array([7,-7,7,-7,5,-5])//np.array([3,3,-3,-3,0,0])");
  if (o) CHECK(allclose(floor_divide(a, b), *o, 0, 0, true));
}

TEST_CASE("mined umath: integer remainder signs + divide-by-zero vs numpy") {
  ndarray a = ival({7, -7, 7, -7, 5, -5});
  ndarray b = ival({3, 3, -3, -3, 0, 0});
  auto o = npt::oracle(
      "a=np.remainder(np.array([7,-7,7,-7,5,-5]),np.array([3,3,-3,-3,0,0]))");
  if (o) CHECK(allclose(mod(a, b), *o, 0, 0, true));
}

TEST_CASE("mined umath: float_power special values vs numpy") {
  ndarray a = dval({0, 1, 2, INF, -1, 0, 2});
  ndarray b = dval({0, INF, -1, 0, 0.5, -1, 10});
  auto o = npt::oracle(
      "a=np.float_power(np.array([0.,1,2,np.inf,-1,0,2]),np.array([0.,np.inf,-1,0,0.5,-1,10]))");
  if (o) CHECK(allclose(float_power(a, b), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined umath: complex sqrt/log/exp branch cuts vs numpy") {
  ndarray z = cval({cd(-1, 0), cd(-1, -0.0), cd(0, 0), cd(1, 1), cd(-4, 0), cd(0, -1)});
  const char* PYZ = "z=np.array([-1+0j,complex(-1,-0.0),0j,1+1j,-4+0j,-1j]); ";
  auto chk = [&](const ndarray& got, const std::string& fn) {
    auto o = npt::oracle(std::string(PYZ) + "a=np." + fn + "(z)");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(sqrt(z), "sqrt");
  chk(log(z), "log");
  chk(exp(z), "exp");
}

TEST_CASE("mined umath: bitwise ufuncs on signed integers vs numpy") {
  ndarray a = ival({5, -5, 12, -12, 0, -1});
  ndarray b = ival({3, 3, 10, 6, 7, 1});
  const char* PYAB = "A=np.array([5,-5,12,-12,0,-1]); B=np.array([3,3,10,6,7,1]); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYAB) + "a=" + e);
    if (o) CHECK(allclose(got, *o, 0, 0, true));
  };
  chk(bitwise_and(a, b), "A & B");
  chk(bitwise_or(a, b), "A | B");
  chk(bitwise_xor(a, b), "A ^ B");
  chk(invert(a), "~A");
  chk(left_shift(a, ival({1, 2, 1, 3, 4, 2})), "A << np.array([1,2,1,3,4,2])");
  chk(right_shift(a, ival({1, 2, 1, 3, 4, 2})), "A >> np.array([1,2,1,3,4,2])");
}

TEST_CASE("mined umath: comparison ufuncs with NaN vs numpy") {
  ndarray a = dval({NAN_, NAN_, 1, 2, NAN_, INF});
  ndarray b = dval({NAN_, 1, NAN_, 2, INF, INF});
  const char* PYAB = "A=np.array([np.nan,np.nan,1,2,np.nan,np.inf]); "
                     "B=np.array([np.nan,1,np.nan,2,np.inf,np.inf]); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYAB) + "a=" + e);
    if (o) CHECK(allclose(got.astype(kFloat64), *o, 0, 0, true));
  };
  chk(equal(a, b), "A == B");
  chk(not_equal(a, b), "A != B");
  chk(less(a, b), "A < B");
  chk(greater(a, b), "A > B");
  chk(less_equal(a, b), "A <= B");
  chk(greater_equal(a, b), "A >= B");
}

TEST_CASE("mined umath: gcd / lcm vs numpy") {
  ndarray a = ival({12, -12, 0, 7, 100});
  ndarray b = ival({18, 8, 5, 0, 80});
  auto og = npt::oracle("a=np.gcd(np.array([12,-12,0,7,100]),np.array([18,8,5,0,80]))");
  auto ol = npt::oracle("a=np.lcm(np.array([12,-12,0,7,100]),np.array([18,8,5,0,80]))");
  if (og) CHECK(allclose(gcd(a, b), *og, 0, 0, true));
  if (ol) CHECK(allclose(lcm(a, b), *ol, 0, 0, true));
}
