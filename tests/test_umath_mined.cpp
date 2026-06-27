// Cases mined from numpy's numpy/_core/tests/test_umath.py: special-value handling
// (+/-inf, nan, +/-0) across the ufuncs, replayed through the live-NumPy oracle.
// Comparisons use equal_nan and treat matching infinities as equal.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace numpp;

namespace {
const double INF = std::numeric_limits<double>::infinity();
const double NAN_ = std::numeric_limits<double>::quiet_NaN();
ndarray dvals(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
// The special-values vector used for unary checks.
ndarray SV() { return dvals({-INF, -2, -1, -0.0, 0.0, 0.5, 1, 2, INF, NAN_}); }
const char* PY_SV = "x=np.array([-np.inf,-2,-1,-0.0,0.0,0.5,1,2,np.inf,np.nan]); ";
}  // namespace

TEST_CASE("mined umath: unary ufuncs on special values vs numpy") {
  ndarray x = SV();
  auto chk = [&](const ndarray& got, const std::string& fn) {
    auto o = npt::oracle(std::string(PY_SV) + "a=np." + fn + "(x)");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(sqrt(x), "sqrt");
  chk(cbrt(x), "cbrt");
  chk(exp(x), "exp");
  chk(expm1(x), "expm1");
  chk(log(x), "log");
  chk(log1p(x), "log1p");
  chk(log2(x), "log2");
  chk(log10(x), "log10");
  chk(reciprocal(x), "reciprocal");
  chk(square(x), "square");
  chk(absolute(x), "absolute");
  chk(negative(x), "negative");
  chk(sign(x), "sign");
  chk(floor(x), "floor");
  chk(ceil(x), "ceil");
  chk(trunc(x), "trunc");
  chk(rint(x), "rint");
  chk(sin(x), "sin");
  chk(cos(x), "cos");
  chk(tanh(x), "tanh");
  chk(arctan(x), "arctan");
}

TEST_CASE("mined umath: sign(nan) is nan, not 0 (regression #81)") {
  ndarray x = dvals({NAN_, -3, -0.0, 0.0, 4});
  auto o = npt::oracle("a=np.sign([np.nan,-3,-0.0,0.0,4.])");
  if (o) CHECK(allclose(sign(x), *o, 0, 0, true));
  CHECK(std::isnan(sign(dvals({NAN_})).item<double>({0})));
}

TEST_CASE("mined umath: rint rounds half to even vs numpy") {
  ndarray x = dvals({-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5});
  auto o = npt::oracle("a=np.rint([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5])");
  if (o) CHECK(allclose(rint(x), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined umath: signbit on signed zeros and infinities vs numpy") {
  ndarray x = dvals({-0.0, 0.0, -INF, INF, -1, 1, NAN_});
  auto o = npt::oracle("a=np.signbit([-0.0,0.0,-np.inf,np.inf,-1,1,np.nan])");
  if (o) CHECK(allclose(signbit(x).astype(kFloat64), *o, 0, 0, true));
}

namespace {
// Two aligned operand vectors for binary special-value checks.
ndarray A2() { return dvals({0, 0, 1, -1, 2, INF, 0, NAN_, INF, -1}); }
ndarray B2() { return dvals({0, -1, INF, 0.5, -1, 0, INF, 2, INF, INF}); }
const char* PY_AB =
    "A=np.array([0.,0,1,-1,2,np.inf,0,np.nan,np.inf,-1]); "
    "B=np.array([0.,-1,np.inf,0.5,-1,0,np.inf,2,np.inf,np.inf]); ";
}  // namespace

TEST_CASE("mined umath: binary ufuncs on special values vs numpy") {
  ndarray a = A2(), b = B2();
  auto chk = [&](const ndarray& got, const std::string& fn) {
    auto o = npt::oracle(std::string(PY_AB) + "a=np." + fn + "(A,B)");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(power(a, b), "power");        // 0**0=1, 0**-1=inf, 1**inf=1, (-1)**.5=nan, inf**inf=inf
  chk(hypot(a, b), "hypot");        // hypot(inf, nan) == inf
  chk(arctan2(a, b), "arctan2");
  chk(copysign(a, b), "copysign");
  chk(maximum(a, b), "maximum");    // nan propagates
  chk(minimum(a, b), "minimum");
  chk(fmax(a, b), "fmax");          // nan ignored
  chk(fmin(a, b), "fmin");
  chk(logaddexp(a, b), "logaddexp");
}

TEST_CASE("mined umath: divide by zero (signs) vs numpy") {
  ndarray a = dvals({1, -1, 0, 2, -2, NAN_});
  ndarray z = dvals({0, 0, 0, 0, 0, 0});
  auto o = npt::oracle("a=np.array([1.,-1,0,2,-2,np.nan])/np.zeros(6)");
  if (o) CHECK(allclose(divide(a, z), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined umath: remainder sign follows the divisor vs numpy") {
  ndarray a = dvals({5, -5, 5, -5, 7, -7});
  ndarray b = dvals({3, 3, -3, -3, 3, -3});
  auto o = npt::oracle(
      "a=np.remainder(np.array([5.,-5,5,-5,7,-7]),np.array([3.,3,-3,-3,3,-3]))");
  if (o) CHECK(allclose(mod(a, b), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined umath: nextafter directions vs numpy") {
  ndarray a = dvals({1, 1, 0, -0.0, INF, 1});
  ndarray b = dvals({2, 0, -1, 1, 0, INF});
  auto o = npt::oracle(
      "a=np.nextafter(np.array([1.,1,0,-0.0,np.inf,1]),np.array([2.,0,-1,1,0,np.inf]))");
  if (o) CHECK(allclose(nextafter(a, b), *o, 0, 0, true));
}
