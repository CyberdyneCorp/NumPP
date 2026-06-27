#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <limits>
#include <vector>

using namespace numpp;

namespace {
ndarray dvals(const std::vector<double>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
const double kInf = std::numeric_limits<double>::infinity();
const double kNan = std::numeric_limits<double>::quiet_NaN();
}  // namespace

TEST_CASE("isclose finite vs numpy") {
  ndarray x = dvals({1.0, 1.0 + 1e-9, 2.0, 3.0});
  ndarray y = dvals({1.0, 1.0, 2.0000001, 3.5});
  if (!npt::numpy_available()) return;
  auto o = npt::oracle("a=np.isclose([1.,1.+1e-9,2.,3.],[1.,1.,2.0000001,3.5])");
  if (!o) return;
  CHECK(allclose(isclose(x, y), *o, 0, 0, true));
}

TEST_CASE("isclose inf and nan vs numpy") {
  ndarray x = dvals({kInf, kInf, -kInf, kNan, kInf, 1.0});
  ndarray y = dvals({kInf, -kInf, -kInf, kNan, 1.0, kNan});
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(
      "a=np.isclose([np.inf,np.inf,-np.inf,np.nan,np.inf,1.],"
      "[np.inf,-np.inf,-np.inf,np.nan,1.,np.nan])");
  if (!o) return;
  CHECK(allclose(isclose(x, y), *o, 0, 0, true));
}

TEST_CASE("isclose equal_nan vs numpy") {
  ndarray x = dvals({kNan, 1.0, kInf});
  ndarray y = dvals({kNan, 1.0, kInf});
  if (!npt::numpy_available()) return;
  auto o = npt::oracle("a=np.isclose([np.nan,1.,np.inf],[np.nan,1.,np.inf],equal_nan=True)");
  if (!o) return;
  CHECK(allclose(isclose(x, y, 1e-5, 1e-8, true), *o, 0, 0, true));
}

TEST_CASE("isclose broadcast vs numpy") {
  ndarray x = dvals({1.0, 2.0, 3.0});
  ndarray s = full({1}, 2.0, kFloat64);
  if (!npt::numpy_available()) return;
  auto o = npt::oracle("a=np.isclose([1.,2.,3.],[2.])");
  if (!o) return;
  CHECK(allclose(isclose(x, s), *o, 0, 0, true));
}

TEST_CASE("isposinf vs numpy") {
  ndarray x = dvals({kInf, -kInf, 1.0, kNan, 0.0});
  if (!npt::numpy_available()) return;
  auto o = npt::oracle("a=np.isposinf([np.inf,-np.inf,1.,np.nan,0.])");
  if (!o) return;
  CHECK(allclose(isposinf(x), *o, 0, 0, true));
}

TEST_CASE("isneginf vs numpy") {
  ndarray x = dvals({kInf, -kInf, 1.0, kNan, 0.0});
  if (!npt::numpy_available()) return;
  auto o = npt::oracle("a=np.isneginf([np.inf,-np.inf,1.,np.nan,0.])");
  if (!o) return;
  CHECK(allclose(isneginf(x), *o, 0, 0, true));
}
