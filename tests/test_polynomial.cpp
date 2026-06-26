#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <vector>

using namespace numpp;
namespace P = numpp::polynomial;

namespace {
ndarray dvals(const std::vector<double>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

// ---- power-basis free functions (lowest-first) ----
TEST_CASE("polynomial.polyval vs numpy") {
  ndarray c = dvals({1, 2, 3});  // 1 + 2x + 3x^2
  ndarray x = dvals({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.polynomial.polyval([0,0.5,1,2],[1,2,3.])");
  if (o) CHECK(allclose(P::polyval(x, c), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polynomial.polyadd vs numpy") {
  ndarray a = dvals({1, 2}), b = dvals({3, 4, 5});
  auto o = npt::oracle("a=np.polynomial.polynomial.polyadd([1,2.],[3,4,5.])");
  if (o) CHECK(allclose(P::polyadd(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polynomial.polysub vs numpy") {
  ndarray a = dvals({1, 2, 3}), b = dvals({1, 1});
  auto o = npt::oracle("a=np.polynomial.polynomial.polysub([1,2,3.],[1,1.])");
  if (o) CHECK(allclose(P::polysub(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polynomial.polymul vs numpy") {
  ndarray a = dvals({1, 2}), b = dvals({1, 0, 3});
  auto o = npt::oracle("a=np.polynomial.polynomial.polymul([1,2.],[1,0,3.])");
  if (o) CHECK(allclose(P::polymul(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polynomial.polyder vs numpy") {
  ndarray c = dvals({1, 2, 3, 4});
  auto o = npt::oracle("a=np.polynomial.polynomial.polyder([1,2,3,4.])");
  if (o) CHECK(allclose(P::polyder(c), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polynomial.polyint vs numpy") {
  ndarray c = dvals({1, 2, 3});
  auto o = npt::oracle("a=np.polynomial.polynomial.polyint([1,2,3.],k=5)");
  if (o) CHECK(allclose(P::polyint(c, 1, 5.0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("polynomial.polyroots sorted vs numpy") {
  ndarray c = dvals({-6, 11, -6, 1});  // (x-1)(x-2)(x-3)
  auto o = npt::oracle("a=np.sort(np.polynomial.polynomial.polyroots([-6,11,-6,1.]).real)");
  if (o) CHECK(allclose(sort(P::polyroots(c)), *o, 1e-6, 1e-9, true));
}
TEST_CASE("polynomial.polyfit vs numpy") {
  ndarray x = arange(0., 8., 1., kFloat64);
  ndarray c = dvals({1, -3, 2});  // 1 - 3x + 2x^2 (lowest-first)
  ndarray y = P::polyval(x, c);
  auto o = npt::oracle(
      "x=np.arange(8.); y=np.polynomial.polynomial.polyval(x,[1,-3,2.]); "
      "a=np.polynomial.polynomial.polyfit(x,y,2)");
  if (o) CHECK(allclose(P::polyfit(x, y, 2), *o, 1e-6, 1e-8, true));
}

// ---- orthogonal basis evaluation ----
TEST_CASE("chebval vs numpy") {
  ndarray c = dvals({1, 2, 3}), x = dvals({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.chebyshev.chebval([0,0.5,1,2],[1,2,3.])");
  if (o) CHECK(allclose(P::chebval(x, c), *o, 1e-9, 1e-12, true));
}
TEST_CASE("legval vs numpy") {
  ndarray c = dvals({1, 2, 3}), x = dvals({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.legendre.legval([0,0.5,1,2],[1,2,3.])");
  if (o) CHECK(allclose(P::legval(x, c), *o, 1e-9, 1e-12, true));
}
TEST_CASE("hermval vs numpy") {
  ndarray c = dvals({1, 2, 3}), x = dvals({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.hermite.hermval([0,0.5,1,2],[1,2,3.])");
  if (o) CHECK(allclose(P::hermval(x, c), *o, 1e-9, 1e-12, true));
}
TEST_CASE("hermeval vs numpy") {
  ndarray c = dvals({1, 2, 3}), x = dvals({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.hermite_e.hermeval([0,0.5,1,2],[1,2,3.])");
  if (o) CHECK(allclose(P::hermeval(x, c), *o, 1e-9, 1e-12, true));
}
TEST_CASE("lagval vs numpy") {
  ndarray c = dvals({1, 2, 3}), x = dvals({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.laguerre.lagval([0,0.5,1,2],[1,2,3.])");
  if (o) CHECK(allclose(P::lagval(x, c), *o, 1e-9, 1e-12, true));
}

// ---- Polynomial class ----
TEST_CASE("Polynomial eval vs numpy") {
  P::Polynomial p(dvals({2, -1, 0, 1}));  // 2 - x + x^3
  ndarray x = dvals({-1, 0, 1, 2});
  auto o = npt::oracle("a=np.polynomial.Polynomial([2,-1,0,1.])(np.array([-1,0,1,2.]))");
  if (o) CHECK(allclose(p(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("Polynomial deriv coef vs numpy") {
  P::Polynomial p(dvals({2, -1, 0, 1}));
  auto o = npt::oracle("a=np.polynomial.Polynomial([2,-1,0,1.]).deriv().coef");
  if (o) CHECK(allclose(p.deriv().coef(), *o, 1e-9, 1e-12, true));
}
TEST_CASE("Polynomial mul coef vs numpy") {
  P::Polynomial a(dvals({1, 2})), b(dvals({1, 0, 3}));
  auto o = npt::oracle("a=(np.polynomial.Polynomial([1,2.])*np.polynomial.Polynomial([1,0,3.])).coef");
  if (o) CHECK(allclose((a * b).coef(), *o, 1e-9, 1e-12, true));
}
TEST_CASE("Polynomial fit vs numpy free polyfit") {
  ndarray x = arange(0., 6., 1., kFloat64);
  ndarray c = dvals({1, 2, -1});
  ndarray y = P::polyval(x, c);
  auto o = npt::oracle(
      "x=np.arange(6.); y=np.polynomial.polynomial.polyval(x,[1,2,-1.]); "
      "a=np.polynomial.polynomial.polyfit(x,y,2)");
  if (o) CHECK(allclose(P::Polynomial::fit(x, y, 2).coef(), *o, 1e-6, 1e-8, true));
}
