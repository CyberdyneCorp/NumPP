// Cases mined from numpy's numpy/lib/tests/test_polynomial.py: roots (real and
// complex-conjugate), polyval, polyfit, polyder/polyint, poly (roots->coef),
// polyadd/polymul, and vander — replayed through the live-NumPy oracle.
// roots are compared after a sort-by-(real,imag) to remove ordering ambiguity.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <algorithm>
#include <complex>
#include <string>
#include <vector>

using namespace numpp;

namespace {
using cd = std::complex<double>;
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
// Sort a complex128 1-D array by (real, imag) — mirrors numpy.sortc.
ndarray sortc(const ndarray& r) {
  ndarray c = r.astype(kComplex128).copy();
  const int64_t n = c.size();
  std::vector<cd> v(c.typed_data<cd>(), c.typed_data<cd>() + n);
  std::sort(v.begin(), v.end(), [](const cd& a, const cd& b) {
    return a.real() != b.real() ? a.real() < b.real() : a.imag() < b.imag();
  });
  ndarray out(Shape{n}, kComplex128, Order::C);
  for (int64_t i = 0; i < n; ++i) out.typed_data<cd>()[i] = v[i];
  return out;
}
}  // namespace

TEST_CASE("mined poly: roots of real and complex-conjugate polynomials vs numpy") {
  ndarray p1 = dval({1, -3, 2});  // (x-1)(x-2)
  auto o1 = npt::oracle("a=np.sort_complex(np.roots([1,-3,2.]).astype(complex))");
  if (o1) CHECK(allclose(sortc(roots(p1)), *o1, 1e-9, 1e-11, true));

  ndarray p2 = dval({1, 0, 1});  // x^2+1 -> +/- i
  auto o2 = npt::oracle("a=np.sort_complex(np.roots([1,0,1.]).astype(complex))");
  if (o2) CHECK(allclose(sortc(roots(p2)), *o2, 1e-9, 1e-11, true));

  ndarray p3 = dval({1, -6, 11, -6});  // (x-1)(x-2)(x-3)
  auto o3 = npt::oracle("a=np.sort_complex(np.roots([1,-6,11,-6.]).astype(complex))");
  if (o3) CHECK(allclose(sortc(roots(p3)), *o3, 1e-8, 1e-10, true));
}

TEST_CASE("mined poly: polyval vs numpy") {
  ndarray p = dval({1, -3, 2});
  ndarray x = dval({0, 1, 2, 3});
  auto o = npt::oracle("a=np.polyval([1,-3,2.],[0,1,2,3.])");
  if (o) CHECK(allclose(polyval(p, x), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined poly: polyfit exact linear / quadratic vs numpy") {
  ndarray x = dval({0, 1, 2, 3});
  ndarray y1 = dval({1, 3, 5, 7});  // y = 2x+1
  auto o1 = npt::oracle("a=np.polyfit([0,1,2,3.],[1,3,5,7.],1)");
  if (o1) CHECK(allclose(polyfit(x, y1, 1), *o1, 1e-9, 1e-9, true));

  ndarray xq = dval({-2, -1, 0, 1, 2});
  ndarray yq = dval({4, 1, 0, 1, 4});  // y = x^2
  auto o2 = npt::oracle("a=np.polyfit([-2,-1,0,1,2.],[4,1,0,1,4.],2)");
  if (o2) CHECK(allclose(polyfit(xq, yq, 2), *o2, 1e-8, 1e-9, true));
}

TEST_CASE("mined poly: polyder / polyint round trip vs numpy") {
  ndarray p = dval({1, 2, 3, 4});
  auto od = npt::oracle("a=np.polyder([1,2,3,4.])");
  if (od) CHECK(allclose(polyder(p), *od, 1e-12, 1e-12, true));
  ndarray q = dval({3, 2, 1});
  auto oi = npt::oracle("a=np.polyint([3,2,1.])");
  if (oi) CHECK(allclose(polyint(q), *oi, 1e-12, 1e-12, true));
}

TEST_CASE("mined poly: poly (roots->coef), polyadd, polymul vs numpy") {
  ndarray r = dval({1, 2, 3});
  auto op = npt::oracle("a=np.poly([1,2,3.])");
  if (op) CHECK(allclose(poly(r), *op, 1e-9, 1e-11, true));

  ndarray a = dval({1, 2}), b = dval({1, 0, 3});
  auto oa = npt::oracle("a=np.polyadd([1,2.],[1,0,3.])");
  if (oa) CHECK(allclose(polyadd(a, b), *oa, 1e-12, 1e-12, true));
  auto om = npt::oracle("a=np.polymul([1,2.],[1,0,3.])");
  if (om) CHECK(allclose(polymul(a, b), *om, 1e-12, 1e-12, true));
}

TEST_CASE("mined poly: vander (decreasing and increasing) vs numpy") {
  ndarray x = dval({1, 2, 3});
  auto od = npt::oracle("a=np.vander([1,2,3.],3)");
  if (od) CHECK(allclose(vander(x, 3), *od, 1e-12, 1e-12, true));
  auto oi = npt::oracle("a=np.vander([1,2,3.],4,increasing=True)");
  if (oi) CHECK(allclose(vander(x, 4, true), *oi, 1e-12, 1e-12, true));
}
