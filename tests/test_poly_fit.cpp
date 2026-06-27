#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <vector>

using namespace numpp;
using namespace numpp::polynomial;

namespace {
// y = 1 + 2x - 0.5 x^2 + 0.1 x^3, evaluated on the given x (exact double math,
// matching the same formula applied in the numpy oracle).
ndarray cubic_y(const ndarray& x) {
  ndarray xf = x.astype(kFloat64).ravel().copy();
  const int64_t n = xf.size();
  ndarray y(Shape{n}, kFloat64, Order::C);
  const double* xp = xf.typed_data<double>();
  double* yp = y.typed_data<double>();
  for (int64_t i = 0; i < n; ++i) {
    const double v = xp[i];
    yp[i] = 1.0 + 2.0 * v - 0.5 * v * v + 0.1 * v * v * v;
  }
  return y;
}
const char* PY_Y =
    "x=np.linspace(0,3,10); y=1+2*x-0.5*x**2+0.1*x**3; ";
}  // namespace

TEST_CASE("Chebyshev.fit reproduces the data") {
  ndarray x = linspace(0.0, 3.0, 10);
  ndarray y = cubic_y(x);
  Chebyshev p = Chebyshev::fit(x, y, 3);
  CHECK(allclose(p(x), y, 1e-7, 1e-9, true));  // exact-fit cubic
}

TEST_CASE("Chebyshev.fit evaluation matches numpy fit") {
  ndarray x = linspace(0.0, 3.0, 10);
  ndarray y = cubic_y(x);
  Chebyshev p = Chebyshev::fit(x, y, 3);
  auto o = npt::oracle(std::string(PY_Y) +
                       "a=np.polynomial.Chebyshev.fit(x,y,3)(x)");
  if (o) CHECK(allclose(p(x), *o, 1e-7, 1e-9, true));
}

TEST_CASE("Chebyshev.fit domain defaults to the data range") {
  ndarray x = linspace(0.0, 3.0, 10);
  ndarray y = cubic_y(x);
  Chebyshev p = Chebyshev::fit(x, y, 3);
  CHECK(p.domain()[0] == 0.0);
  CHECK(p.domain()[1] == 3.0);
}

TEST_CASE("Legendre.fit evaluation matches numpy fit") {
  ndarray x = linspace(0.0, 3.0, 10);
  ndarray y = cubic_y(x);
  Legendre p = Legendre::fit(x, y, 3);
  auto o = npt::oracle(std::string(PY_Y) +
                       "a=np.polynomial.Legendre.fit(x,y,3)(x)");
  if (o) CHECK(allclose(p(x), *o, 1e-7, 1e-9, true));
}

TEST_CASE("Laguerre.fit evaluation matches numpy fit (positive domain)") {
  ndarray x = linspace(0.5, 4.0, 12);
  ndarray y = cubic_y(x);
  Laguerre p = Laguerre::fit(x, y, 3);
  auto o = npt::oracle(
      "x=np.linspace(0.5,4,12); y=1+2*x-0.5*x**2+0.1*x**3; "
      "a=np.polynomial.Laguerre.fit(x,y,3)(x)");
  if (o) CHECK(allclose(p(x), *o, 1e-6, 1e-8, true));
}

TEST_CASE("Chebyshev domain/window mapping matches numpy") {
  ndarray coef = arange(1.0, 4.0, 1.0);  // [1,2,3]
  Chebyshev p(coef, {0.0, 2.0}, {-1.0, 1.0});
  ndarray x = linspace(0.0, 2.0, 5);
  auto o = npt::oracle(
      "a=np.polynomial.Chebyshev([1,2,3.],domain=[0,2],window=[-1,1])"
      "(np.linspace(0,2,5))");
  if (o) CHECK(allclose(p(x), *o, 1e-9, 1e-12, true));
}
