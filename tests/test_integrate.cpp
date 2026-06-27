#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

// ---- 1-D ----
TEST_CASE("trapezoid 1d default vs numpy") {
  if (!npt::numpy_available()) return;
  ndarray y = linspace(0.0, 10.0, 11);
  auto o = npt::oracle("a=np.trapezoid(np.linspace(0,10,11))");
  if (!o) return;
  CHECK(allclose(trapezoid(y), *o, 1e-9, 1e-12, true));
}
TEST_CASE("trapezoid 1d dx vs numpy") {
  if (!npt::numpy_available()) return;
  ndarray y = linspace(0.0, 10.0, 11);
  auto o = npt::oracle("a=np.trapezoid(np.linspace(0,10,11),dx=0.5)");
  if (!o) return;
  CHECK(allclose(trapezoid(y, 0.5), *o, 1e-9, 1e-12, true));
}
TEST_CASE("trapezoid 1d x vs numpy") {
  if (!npt::numpy_available()) return;
  ndarray y = linspace(0.0, 10.0, 11);
  ndarray x = linspace(0.0, 5.0, 11);
  auto o = npt::oracle(
      "a=np.trapezoid(np.linspace(0,10,11),x=np.linspace(0,5,11))");
  if (!o) return;
  CHECK(allclose(trapezoid(y, x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("trapz alias 1d vs numpy") {
  if (!npt::numpy_available()) return;
  ndarray y = linspace(0.0, 10.0, 11);
  auto o = npt::oracle("a=np.trapezoid(np.linspace(0,10,11),dx=2.0)");
  if (!o) return;
  CHECK(allclose(trapz(y, 2.0), *o, 1e-9, 1e-12, true));
}

// ---- 2-D ----
TEST_CASE("trapezoid 2d axis0 vs numpy") {
  if (!npt::numpy_available()) return;
  ndarray Y = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.trapezoid(np.arange(12.0).reshape(3,4),axis=0)");
  if (!o) return;
  CHECK(allclose(trapezoid(Y, 1.0, 0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("trapezoid 2d axis1 vs numpy") {
  if (!npt::numpy_available()) return;
  ndarray Y = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.trapezoid(np.arange(12.0).reshape(3,4),axis=1)");
  if (!o) return;
  CHECK(allclose(trapezoid(Y, 1.0, 1), *o, 1e-9, 1e-12, true));
}
TEST_CASE("trapezoid 2d axis negative vs numpy") {
  if (!npt::numpy_available()) return;
  ndarray Y = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.trapezoid(np.arange(12.0).reshape(3,4),axis=-1)");
  if (!o) return;
  CHECK(allclose(trapezoid(Y, 1.0, -1), *o, 1e-9, 1e-12, true));
}
