// Cases mined from numpy's numpy/_core/tests/test_numeric.py: cross, tensordot,
// outer/inner, correlate/convolve modes — vs the live oracle.
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
}  // namespace

TEST_CASE("mined numeric: cross of 3-D vectors vs numpy") {
  ndarray a = dval({1, 2, 3}), b = dval({4, 5, 6});
  auto o = npt::oracle("a=np.cross([1,2,3.],[4,5,6.])");
  if (o) CHECK(allclose(cross(a, b), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined numeric: cross of 2-D vectors (z-component) vs numpy") {
  ndarray a = dval({1, 2}), b = dval({3, 4});
  auto o = npt::oracle("import warnings; warnings.simplefilter('ignore'); a=np.array(np.cross([1,2.],[3,4.]))");
  if (o) CHECK(allclose(cross(a, b), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined numeric: batched cross over rows vs numpy") {
  ndarray A = arange(1.0, 7.0, 1.0).reshape({2, 3});
  ndarray B = arange(7.0, 13.0, 1.0).reshape({2, 3});
  auto o = npt::oracle(
      "A=np.arange(1,7.).reshape(2,3); B=np.arange(7,13.).reshape(2,3); a=np.cross(A,B)");
  if (o) CHECK(allclose(cross(A, B), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined numeric: tensordot axis variants vs numpy") {
  ndarray A = arange(0.0, 24.0, 1.0).reshape({2, 3, 4});
  ndarray B = arange(0.0, 24.0, 1.0).reshape({4, 3, 2});
  const char* PYAB = "A=np.arange(24.).reshape(2,3,4); B=np.arange(24.).reshape(4,3,2); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYAB) + "a=" + e);
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  chk(tensordot(A, B, 1), "np.tensordot(A,B,axes=1)");                       // contract A:-1 with B:0
  chk(tensordot(A, B, {2, 1}, {0, 1}), "np.tensordot(A,B,axes=([2,1],[0,1]))");
}

TEST_CASE("mined numeric: outer / inner vs numpy") {
  ndarray a = dval({1, 2, 3}), b = dval({4, 5});
  auto oo = npt::oracle("a=np.outer([1,2,3.],[4,5.])");
  if (oo) CHECK(allclose(outer(a, b), *oo, 1e-12, 1e-12, true));
  ndarray c = dval({1, 2, 3}), d = dval({4, 5, 6});
  auto oi = npt::oracle("a=np.array(np.inner([1,2,3.],[4,5,6.]))");
  if (oi) CHECK(allclose(inner(c, d), *oi, 1e-12, 1e-12, true));
}

TEST_CASE("mined numeric: correlate modes vs numpy") {
  ndarray a = dval({1, 2, 3, 4, 5});
  ndarray v = dval({0, 1, 0.5});
  const char* PYAV = "a0=np.array([1,2,3,4,5.]); v0=np.array([0,1,0.5]); ";
  auto chk = [&](const ndarray& got, const std::string& mode) {
    auto o = npt::oracle(std::string(PYAV) + "a=np.correlate(a0,v0,'" + mode + "')");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(correlate(a, v, "valid"), "valid");
  chk(correlate(a, v, "same"), "same");
  chk(correlate(a, v, "full"), "full");
}

TEST_CASE("mined numeric: convolve modes vs numpy") {
  ndarray a = dval({1, 2, 3});
  ndarray v = dval({0, 1, 0.5});
  const char* PYAV = "a0=np.array([1,2,3.]); v0=np.array([0,1,0.5]); ";
  auto chk = [&](const ndarray& got, const std::string& mode) {
    auto o = npt::oracle(std::string(PYAV) + "a=np.convolve(a0,v0,'" + mode + "')");
    if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
  };
  chk(convolve(a, v, "full"), "full");
  chk(convolve(a, v, "same"), "same");
  chk(convolve(a, v, "valid"), "valid");
}
