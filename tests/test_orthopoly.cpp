#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;
using namespace numpp::polynomial;

namespace OP = numpp::polynomial;

namespace {
ndarray op_dv(const std::vector<double>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

// ---- Vandermonde matrices (all five) ----
TEST_CASE("orthopoly.chebvander vs numpy") {
  ndarray x = op_dv({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.chebyshev.chebvander([0,.5,1,2.],3)");
  if (o) CHECK(allclose(OP::chebvander(x, 3), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.legvander vs numpy") {
  ndarray x = op_dv({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.legendre.legvander([0,.5,1,2.],3)");
  if (o) CHECK(allclose(OP::legvander(x, 3), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.hermvander vs numpy") {
  ndarray x = op_dv({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.hermite.hermvander([0,.5,1,2.],3)");
  if (o) CHECK(allclose(OP::hermvander(x, 3), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.hermevander vs numpy") {
  ndarray x = op_dv({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.hermite_e.hermevander([0,.5,1,2.],3)");
  if (o) CHECK(allclose(OP::hermevander(x, 3), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.lagvander vs numpy") {
  ndarray x = op_dv({0, 0.5, 1, 2});
  auto o = npt::oracle("a=np.polynomial.laguerre.lagvander([0,.5,1,2.],3)");
  if (o) CHECK(allclose(OP::lagvander(x, 3), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.chebvander deg0 column of ones") {
  ndarray x = op_dv({-1, 0, 3.5});
  auto o = npt::oracle("a=np.polynomial.chebyshev.chebvander([-1,0,3.5],0)");
  if (o) CHECK(allclose(OP::chebvander(x, 0), *o, 1e-9, 1e-12, true));
}

// ---- Roots (coeffs chosen to yield the real roots {-0.7,0.2,0.9}) ----
TEST_CASE("orthopoly.chebroots vs numpy") {
  ndarray c = op_dv({-0.074, 0.16, -0.2, 0.25});
  auto o = npt::oracle(
      "a=np.sort(np.polynomial.chebyshev.chebroots([-0.074,0.16,-0.2,0.25]).real)");
  if (o) CHECK(allclose(OP::chebroots(c), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.legroots vs numpy") {
  ndarray c = op_dv({-0.0073333333333333, 0.01, -0.2666666666666667, 0.4});
  auto o = npt::oracle(
      "a=np.sort(np.polynomial.legendre.legroots("
      "[-0.0073333333333333,0.01,-0.2666666666666667,0.4]).real)");
  if (o) CHECK(allclose(OP::legroots(c), *o, 1e-8, 1e-10, true));
}
TEST_CASE("orthopoly.hermroots vs numpy") {
  ndarray c = op_dv({-0.074, 0.455, -0.1, 0.125});
  auto o = npt::oracle(
      "a=np.sort(np.polynomial.hermite.hermroots([-0.074,0.455,-0.1,0.125]).real)");
  if (o) CHECK(allclose(OP::hermroots(c), *o, 1e-8, 1e-10, true));
}
TEST_CASE("orthopoly.hermeroots vs numpy") {
  ndarray c = op_dv({-0.274, 2.41, -0.4, 1.0});
  auto o = npt::oracle(
      "a=np.sort(np.polynomial.hermite_e.hermeroots([-0.274,2.41,-0.4,1.0]).real)");
  if (o) CHECK(allclose(OP::hermeroots(c), *o, 1e-8, 1e-10, true));
}
TEST_CASE("orthopoly.lagroots vs numpy") {
  ndarray c = op_dv({4.736, -15.81, 17.2, -6.0});
  auto o = npt::oracle(
      "a=np.sort(np.polynomial.laguerre.lagroots([4.736,-15.81,17.2,-6.0]).real)");
  if (o) CHECK(allclose(OP::lagroots(c), *o, 1e-8, 1e-10, true));
}
TEST_CASE("orthopoly.chebroots linear single root") {
  ndarray c = op_dv({0.6, 0.25});  // 0.6 + 0.25*T1 -> root -2.4
  auto o = npt::oracle("a=np.sort(np.polynomial.chebyshev.chebroots([0.6,0.25]).real)");
  if (o) CHECK(allclose(OP::chebroots(c), *o, 1e-9, 1e-12, true));
}
