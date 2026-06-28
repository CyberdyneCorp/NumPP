// Regression tests for #11: array printing switches to scientific notation for
// large/small magnitudes, matching numpy's str() byte-for-byte (incl. exponent
// alignment, fractional-width padding, specials, and per-part complex columns).
#include <cstdio>
#include <complex>
#include <string>
#include <vector>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

namespace {
using cd = std::complex<double>;
ndarray dvec(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray cvec(const std::vector<cd>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kComplex128, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<cd>()[i] = v[i];
  return a;
}
std::string py_str(const std::string& a) {
  std::string cmd = "import numpy as np,sys; a=" + a + "; sys.stdout.write(str(a))";
  std::string out;
  if (FILE* f = popen(("python3 -c \"" + cmd + "\"").c_str(), "r")) {
    char b[8192]; size_t n;
    while ((n = fread(b, 1, sizeof(b), f)) > 0) out.append(b, n);
    pclose(f);
  }
  return out;
}
void chk(const ndarray& got, const std::string& npexpr, const char* label) {
  if (!npt::numpy_available()) { std::fprintf(stderr, "  [skip] %s\n", label); return; }
  std::string want = py_str(npexpr);
  if (array_str(got) != want)
    std::fprintf(stderr, "  mismatch %s:\n  got : [%s]\n  want: [%s]\n", label, array_str(got).c_str(), want.c_str());
  CHECK(array_str(got) == want);
}
const double INF = std::numeric_limits<double>::infinity();
const double NAN_ = std::numeric_limits<double>::quiet_NaN();
}  // namespace

TEST_CASE("print sci: large/small magnitudes switch to scientific vs numpy") {
  chk(dvec({1e8, 2e8, 3e8}), "np.array([1e8,2e8,3e8])", "big");
  chk(dvec({1e-5, 2e-5}), "np.array([1e-5,2e-5])", "small");
  chk(dvec({123456789.0}), "np.array([123456789.0])", "full-mantissa");
  chk(dvec({1.5e8, 2.5e-3}), "np.array([1.5e8,2.5e-3])", "wide-range");
  chk(dvec({0.0, 1e9}), "np.array([0.0,1e9])", "zero-and-big");
}

TEST_CASE("print sci: trigger boundary (ratio > 1000) vs numpy") {
  chk(dvec({1.0, 1001.0}), "np.array([1.0,1001.0])", "ratio-1001-sci");
  chk(dvec({1.0, 1000.0}), "np.array([1.0,1000.0])", "ratio-1000-fixed");
  chk(dvec({1.5, 2.25, 3.0}), "np.array([1.5,2.25,3.0])", "ordinary-fixed");
}

TEST_CASE("print sci: alignment, signs, specials, exponent width vs numpy") {
  chk(dvec({-1.5e8, 2.5e8}), "np.array([-1.5e8,2.5e8])", "negatives");
  chk(dvec({1.25e8, 2.0e8}), "np.array([1.25e8,2.0e8])", "fractional-pad");
  chk(dvec({1e8, INF, -INF}), "np.array([1e8,np.inf,-np.inf])", "infinities");
  chk(dvec({1e9, NAN_}), "np.array([1e9,np.nan])", "nan");
  chk(dvec({0.0, 1.25e8}), "np.array([0.0,1.25e8])", "zero-fractional-pad");
  chk(dvec({1e100, 1e-100}), "np.array([1e100,1e-100])", "three-digit-exponent");
}

TEST_CASE("print sci: 2-D and float32 vs numpy") {
  ndarray m = dvec({1e8, 2e8, 3e8, 4e8}).reshape({2, 2});
  chk(m, "np.array([1e8,2e8,3e8,4e8]).reshape(2,2)", "2d");
  chk(dvec({1e8, 2e8}).astype(kFloat32), "np.array([1e8,2e8],dtype=np.float32)", "float32");
}

TEST_CASE("print sci: complex parts format independently vs numpy") {
  chk(cvec({cd(1e8, 2e8), cd(3e8, 4e8)}), "np.array([1e8+2e8j,3e8+4e8j])", "complex-big");
  chk(cvec({cd(1e-5, 2e-5)}), "np.array([1e-5+2e-5j])", "complex-small");
  chk(cvec({cd(1.5e8, -2.5e-3)}), "np.array([1.5e8-2.5e-3j])", "complex-mixed-parts");
  chk(cvec({cd(1, 2), cd(3, -4)}), "np.array([1+2j,3-4j])", "complex-ordinary-fixed");
}
