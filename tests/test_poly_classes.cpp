#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;
using namespace numpp::polynomial;

namespace {
ndarray pc_dvals(const std::vector<double>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, numpp::kFloat64, numpp::Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray pc_sorted(const ndarray& in) {
  ndarray a = in.astype(numpp::kFloat64).ravel().copy();
  int64_t n = a.size();
  double* d = a.typed_data<double>();
  std::sort(d, d + n);
  return a;
}
}  // namespace

// ---- Chebyshev ----
TEST_CASE("poly-classes Chebyshev eval vs numpy") {
  numpp::polynomial::Chebyshev p(pc_dvals({1, 2, 3}));
  ndarray x = pc_dvals({-1, 0, 0.5, 1});
  auto o = npt::oracle("a=np.polynomial.Chebyshev([1,2,3.])(np.array([-1,0,.5,1.]))");
  if (o) CHECK(allclose(p(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes Chebyshev deriv vs numpy") {
  numpp::polynomial::Chebyshev p(pc_dvals({1, 2, 3, 4}));
  auto o = npt::oracle("a=np.polynomial.Chebyshev([1,2,3,4.]).deriv().coef");
  if (o) CHECK(allclose(p.deriv().coef(), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes Chebyshev integ vs numpy") {
  numpp::polynomial::Chebyshev p(pc_dvals({1, 2, 3}));
  auto o = npt::oracle("a=np.polynomial.Chebyshev([1,2,3.]).integ(1,k=5).coef");
  if (o) CHECK(allclose(p.integ(1, 5.0).coef(), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes Chebyshev roots vs numpy") {
  numpp::polynomial::Chebyshev p(pc_dvals({0, 0, 1}));
  auto o = npt::oracle("a=np.sort(np.polynomial.Chebyshev([0,0,1.]).roots())");
  if (o) CHECK(allclose(pc_sorted(p.roots()), *o, 1e-9, 1e-12, true));
}

// ---- Legendre ----
TEST_CASE("poly-classes Legendre eval vs numpy") {
  numpp::polynomial::Legendre p(pc_dvals({1, 2, 3}));
  ndarray x = pc_dvals({-1, 0, 0.5, 1});
  auto o = npt::oracle("a=np.polynomial.Legendre([1,2,3.])(np.array([-1,0,.5,1.]))");
  if (o) CHECK(allclose(p(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes Legendre integ vs numpy") {
  numpp::polynomial::Legendre p(pc_dvals({1, 2, 3}));
  auto o = npt::oracle("a=np.polynomial.Legendre([1,2,3.]).integ(1,k=5).coef");
  if (o) CHECK(allclose(p.integ(1, 5.0).coef(), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes Legendre roots vs numpy") {
  numpp::polynomial::Legendre p(pc_dvals({0, 0, 1}));
  auto o = npt::oracle("a=np.sort(np.polynomial.Legendre([0,0,1.]).roots())");
  if (o) CHECK(allclose(pc_sorted(p.roots()), *o, 1e-9, 1e-12, true));
}

// ---- Hermite ----
TEST_CASE("poly-classes Hermite eval vs numpy") {
  numpp::polynomial::Hermite p(pc_dvals({1, 2, 3}));
  ndarray x = pc_dvals({-1, 0, 0.5, 1});
  auto o = npt::oracle("a=np.polynomial.Hermite([1,2,3.])(np.array([-1,0,.5,1.]))");
  if (o) CHECK(allclose(p(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes Hermite deriv vs numpy") {
  numpp::polynomial::Hermite p(pc_dvals({1, 2, 3, 4}));
  auto o = npt::oracle("a=np.polynomial.Hermite([1,2,3,4.]).deriv().coef");
  if (o) CHECK(allclose(p.deriv().coef(), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes Hermite roots vs numpy") {
  numpp::polynomial::Hermite p(pc_dvals({0, 0, 1}));
  auto o = npt::oracle("a=np.sort(np.polynomial.Hermite([0,0,1.]).roots())");
  if (o) CHECK(allclose(pc_sorted(p.roots()), *o, 1e-9, 1e-12, true));
}

// ---- HermiteE ----
TEST_CASE("poly-classes HermiteE eval vs numpy") {
  numpp::polynomial::HermiteE p(pc_dvals({1, 2, 3}));
  ndarray x = pc_dvals({-1, 0, 0.5, 1});
  auto o = npt::oracle("a=np.polynomial.HermiteE([1,2,3.])(np.array([-1,0,.5,1.]))");
  if (o) CHECK(allclose(p(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes HermiteE integ vs numpy") {
  numpp::polynomial::HermiteE p(pc_dvals({1, 2, 3}));
  auto o = npt::oracle("a=np.polynomial.HermiteE([1,2,3.]).integ(1,k=5).coef");
  if (o) CHECK(allclose(p.integ(1, 5.0).coef(), *o, 1e-9, 1e-12, true));
}

// ---- Laguerre ----
TEST_CASE("poly-classes Laguerre eval vs numpy") {
  numpp::polynomial::Laguerre p(pc_dvals({1, 2, 3}));
  ndarray x = pc_dvals({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.Laguerre([1,2,3.])(np.array([0,.5,1,2.]))");
  if (o) CHECK(allclose(p(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes Laguerre deriv vs numpy") {
  numpp::polynomial::Laguerre p(pc_dvals({1, 2, 3, 4}));
  auto o = npt::oracle("a=np.polynomial.Laguerre([1,2,3,4.]).deriv().coef");
  if (o) CHECK(allclose(p.deriv().coef(), *o, 1e-9, 1e-12, true));
}
TEST_CASE("poly-classes Laguerre roots vs numpy") {
  numpp::polynomial::Laguerre p(pc_dvals({1, -2, 1}));
  auto o = npt::oracle("a=np.sort(np.polynomial.Laguerre([1,-2,1.]).roots())");
  if (o) CHECK(allclose(pc_sorted(p.roots()), *o, 1e-9, 1e-12, true));
}
