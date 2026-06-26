#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;
using namespace numpp::polynomial;

namespace OPDI = numpp::polynomial;

namespace {
ndarray opdi_v(const std::vector<double>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("orthopoly.chebder vs numpy") {
  auto o = npt::oracle("a=np.polynomial.chebyshev.chebder([1,2,3,4.])");
  if (o) CHECK(allclose(OPDI::chebder(opdi_v({1, 2, 3, 4})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.chebint vs numpy") {
  auto o = npt::oracle("a=np.polynomial.chebyshev.chebint([1,2,3.],k=5)");
  if (o) CHECK(allclose(OPDI::chebint(opdi_v({1, 2, 3}), 1, 5.0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.legder vs numpy") {
  auto o = npt::oracle("a=np.polynomial.legendre.legder([1,2,3,4.])");
  if (o) CHECK(allclose(OPDI::legder(opdi_v({1, 2, 3, 4})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.legint vs numpy") {
  auto o = npt::oracle("a=np.polynomial.legendre.legint([1,2,3.],k=5)");
  if (o) CHECK(allclose(OPDI::legint(opdi_v({1, 2, 3}), 1, 5.0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.hermder vs numpy") {
  auto o = npt::oracle("a=np.polynomial.hermite.hermder([1,2,3,4.])");
  if (o) CHECK(allclose(OPDI::hermder(opdi_v({1, 2, 3, 4})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.hermint vs numpy") {
  auto o = npt::oracle("a=np.polynomial.hermite.hermint([1,2,3.],k=5)");
  if (o) CHECK(allclose(OPDI::hermint(opdi_v({1, 2, 3}), 1, 5.0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.hermeder vs numpy") {
  auto o = npt::oracle("a=np.polynomial.hermite_e.hermeder([1,2,3,4.])");
  if (o) CHECK(allclose(OPDI::hermeder(opdi_v({1, 2, 3, 4})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.hermeint vs numpy") {
  auto o = npt::oracle("a=np.polynomial.hermite_e.hermeint([1,2,3.],k=5)");
  if (o) CHECK(allclose(OPDI::hermeint(opdi_v({1, 2, 3}), 1, 5.0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.lagder vs numpy") {
  auto o = npt::oracle("a=np.polynomial.laguerre.lagder([1,2,3,4.])");
  if (o) CHECK(allclose(OPDI::lagder(opdi_v({1, 2, 3, 4})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.lagint vs numpy") {
  auto o = npt::oracle("a=np.polynomial.laguerre.lagint([1,2,3.],k=5)");
  if (o) CHECK(allclose(OPDI::lagint(opdi_v({1, 2, 3}), 1, 5.0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.chebder m=2 vs numpy") {
  auto o = npt::oracle("a=np.polynomial.chebyshev.chebder([1,2,3,4,5.],2)");
  if (o) CHECK(allclose(OPDI::chebder(opdi_v({1, 2, 3, 4, 5}), 2), *o, 1e-9, 1e-12, true));
}
TEST_CASE("orthopoly.legint m=2 vs numpy") {
  auto o = npt::oracle("a=np.polynomial.legendre.legint([1,2,3.],2,k=1)");
  if (o) CHECK(allclose(OPDI::legint(opdi_v({1, 2, 3}), 2, 1.0), *o, 1e-9, 1e-12, true));
}
