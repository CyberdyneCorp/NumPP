// Cases mined from numpy's numpy/lib/tests/test_type_check.py: nan_to_num (default
// and custom nan/posinf/neginf fills), angle, isreal/iscomplex, real_if_close.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <complex>
#include <limits>
#include <string>
#include <vector>

using namespace numpp;

namespace {
using cd = std::complex<double>;
const double INF = std::numeric_limits<double>::infinity();
const double NAN_ = std::numeric_limits<double>::quiet_NaN();
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray cvec(const std::vector<cd>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kComplex128, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<cd>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined type_check: nan_to_num default fills vs numpy") {
  ndarray a = dval({NAN_, INF, -INF, 1.5});
  auto o = npt::oracle("a=np.nan_to_num([np.nan,np.inf,-np.inf,1.5])");
  if (o) CHECK(allclose(nan_to_num(a), *o, 0, 0, true));
}

TEST_CASE("mined type_check: nan_to_num custom nan/posinf/neginf vs numpy") {
  ndarray a = dval({NAN_, INF, -INF, 1.5});
  auto o = npt::oracle("a=np.nan_to_num([np.nan,np.inf,-np.inf,1.5],nan=-1,posinf=100,neginf=-100)");
  if (o) CHECK(allclose(nan_to_num(a, -1.0, 100.0, -100.0), *o, 0, 0, true));
}

TEST_CASE("mined type_check: angle of complex values vs numpy") {
  ndarray a = cvec({cd(1, 1), cd(1, -1), cd(-1, 0), cd(0, 2)});
  auto o = npt::oracle("a=np.angle([1+1j,1-1j,-1+0j,0+2j])");
  if (o) CHECK(allclose(angle(a), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined type_check: isreal / iscomplex masks vs numpy") {
  ndarray a = cvec({cd(1, 0), cd(0, 1), cd(2, 0), cd(3, -4)});
  auto oi = npt::oracle("a=np.isreal([1+0j,0+1j,2+0j,3-4j])");
  if (oi) CHECK(allclose(isreal(a).astype(kFloat64), *oi, 0, 0, true));
  auto oc = npt::oracle("a=np.iscomplex([1+0j,0+1j,2+0j,3-4j])");
  if (oc) CHECK(allclose(iscomplex(a).astype(kFloat64), *oc, 0, 0, true));
}

TEST_CASE("mined type_check: real_if_close drops tiny imaginary parts vs numpy") {
  ndarray a = cvec({cd(1, 1e-15), cd(2, 2e-15), cd(3, 0)});
  auto o = npt::oracle("a=np.real_if_close([1+1e-15j,2+2e-15j,3+0j])");
  if (o) CHECK(allclose(real_if_close(a), *o, 1e-12, 1e-12, true));
}
