// Deeper test_umath.py mining: real inverse-trig/hyperbolic domain edges,
// complex branch cuts (trig/inverse/hyperbolic, conj/angle/abs/sign), around/fix,
// logaddexp2 — all vs the live oracle with equal_nan + inf-aware comparison.
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

TEST_CASE("mined umath: real inverse-trig/hyperbolic domain edges vs numpy") {
  // spans below/at/inside/at/above the [-1,1] and [1,inf) domains
  ndarray x = dval({-2, -1, -0.5, 0, 0.5, 1, 2, INF, NAN_});
  const char* PYX = "x=np.array([-2,-1,-0.5,0,0.5,1,2,np.inf,np.nan]); ";
  auto chk = [&](const ndarray& got, const std::string& fn) {
    auto o = npt::oracle(std::string(PYX) + "a=np." + fn + "(x)");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(arcsin(x), "arcsin");    // nan outside [-1,1]
  chk(arccos(x), "arccos");
  chk(arctan(x), "arctan");
  chk(arctanh(x), "arctanh");  // +/-inf at +/-1, nan outside
  chk(arcsinh(x), "arcsinh");
  chk(arccosh(x), "arccosh");  // nan below 1
  chk(tan(x), "tan");
  chk(sinh(x), "sinh");
  chk(cosh(x), "cosh");
}

TEST_CASE("mined umath: around (decimals) and fix vs numpy") {
  ndarray x = dval({-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 1.25, 1.35, -1.25});
  const char* PYX = "x=np.array([-2.5,-1.5,-0.5,0.5,1.5,2.5,1.25,1.35,-1.25]); ";
  auto o0 = npt::oracle(std::string(PYX) + "a=np.round(x,0)");
  auto o1 = npt::oracle(std::string(PYX) + "a=np.round(x,1)");
  auto of = npt::oracle(std::string(PYX) + "a=np.fix(x)");
  if (o0) CHECK(allclose(around(x, 0), *o0, 1e-12, 1e-12, true));
  if (o1) CHECK(allclose(around(x, 1), *o1, 1e-12, 1e-12, true));
  if (of) CHECK(allclose(fix(x), *of, 1e-12, 1e-12, true));
}

TEST_CASE("mined umath: logaddexp2 on special values vs numpy") {
  ndarray a = dval({1, -INF, INF, 0, -INF});
  ndarray b = dval({2, 3, INF, 0, -INF});
  auto o = npt::oracle(
      "a=np.logaddexp2(np.array([1.,-np.inf,np.inf,0,-np.inf]),np.array([2.,3,np.inf,0,-np.inf]))");
  if (o) CHECK(allclose(logaddexp2(a, b), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined umath: complex unary functions vs numpy") {
  // Generic (off-branch-cut) complex values: the transcendentals/inverses agree
  // across implementations only away from their branch cuts and singularities;
  // on-cut / signed-zero points are implementation-defined (numpy itself xfails
  // those on some platforms), so they are intentionally excluded here.
  ndarray z = cval({cd(1, 1), cd(2, 3), cd(-2, -3), cd(0.5, 0.5), cd(-1.5, 2), cd(3, -1)});
  const char* PYZ = "z=np.array([1+1j,2+3j,-2-3j,0.5+0.5j,-1.5+2j,3-1j]); ";
  auto chk = [&](const ndarray& got, const std::string& fn) {
    auto o = npt::oracle(std::string(PYZ) + "a=np." + fn + "(z)");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(sin(z), "sin");
  chk(cos(z), "cos");
  chk(tan(z), "tan");
  chk(arctan(z), "arctan");
  chk(arcsinh(z), "arcsinh");
  chk(arctanh(z), "arctanh");
  chk(sqrt(z), "sqrt");
  chk(conj(z), "conj");
}

TEST_CASE("mined umath: complex angle / abs / sign vs numpy") {
  ndarray z = cval({cd(1, 0), cd(-1, 0), cd(0, 1), cd(0, -1), cd(3, 4), cd(-3, -4)});
  const char* PYZ = "z=np.array([1+0j,-1+0j,1j,-1j,3+4j,-3-4j]); ";
  auto oa = npt::oracle(std::string(PYZ) + "a=np.angle(z)");
  auto ob = npt::oracle(std::string(PYZ) + "a=np.abs(z)");
  auto os = npt::oracle(std::string(PYZ) + "a=np.sign(z)");
  if (oa) CHECK(allclose(angle(z), *oa, 1e-12, 1e-12, true));
  if (ob) CHECK(allclose(absolute(z), *ob, 1e-12, 1e-12, true));
  if (os) CHECK(allclose(sign(z), *os, 1e-12, 1e-12, true));
}
