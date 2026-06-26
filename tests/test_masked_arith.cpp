#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <vector>

using namespace numpp;

// Build a 1-D float64 array from values.
static numpp::ndarray ma_vec(std::vector<double> v) {
  using namespace numpp;
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64);
  for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) a.set_item<double>({i}, v[i]);
  return a;
}
// Build a 1-D bool mask from 0/1 values.
static numpp::ndarray ma_mask(std::vector<int> v) {
  using namespace numpp;
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kBool);
  for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) a.set_item<bool>({i}, v[i] != 0);
  return a;
}

TEST_CASE("ma add vs numpy") {
  using namespace numpp; namespace ma = numpp::ma;
  ma::MaskedArray A = ma::masked_array(ma_vec({1, 2, 3, 4}), ma_mask({0, 1, 0, 0}));
  ma::MaskedArray B = ma::masked_array(ma_vec({5, 6, 7, 8}), ma_mask({0, 0, 1, 0}));
  ma::MaskedArray R = ma::add(A, B);
  auto o = npt::oracle(
      "A=np.ma.masked_array([1,2,3,4.],mask=[0,1,0,0]);"
      "B=np.ma.masked_array([5,6,7,8.],mask=[0,0,1,0]);a=(A+B).filled(-1.)");
  if (o) CHECK(allclose(ma::filled(R, -1.0), *o, 1e-9, 1e-12, true));
  auto om = npt::oracle(
      "A=np.ma.masked_array([1,2,3,4.],mask=[0,1,0,0]);"
      "B=np.ma.masked_array([5,6,7,8.],mask=[0,0,1,0]);"
      "a=np.ma.getmaskarray(A+B).astype(np.float64)");
  if (om) CHECK(allclose(ma::getmask(R).astype(kFloat64), *om, 0, 0, true));
}

TEST_CASE("ma subtract/multiply vs numpy") {
  using namespace numpp; namespace ma = numpp::ma;
  ma::MaskedArray A = ma::masked_array(ma_vec({10, 20, 30, 40}), ma_mask({1, 0, 0, 0}));
  ma::MaskedArray B = ma::masked_array(ma_vec({1, 2, 3, 4}), ma_mask({0, 0, 1, 0}));
  ma::MaskedArray S = ma::subtract(A, B);
  ma::MaskedArray M = ma::multiply(A, B);
  auto os = npt::oracle(
      "A=np.ma.masked_array([10,20,30,40.],mask=[1,0,0,0]);"
      "B=np.ma.masked_array([1,2,3,4.],mask=[0,0,1,0]);a=(A-B).filled(-1.)");
  if (os) CHECK(allclose(ma::filled(S, -1.0), *os, 1e-9, 1e-12, true));
  auto om = npt::oracle(
      "A=np.ma.masked_array([10,20,30,40.],mask=[1,0,0,0]);"
      "B=np.ma.masked_array([1,2,3,4.],mask=[0,0,1,0]);a=(A*B).filled(-1.)");
  if (om) CHECK(allclose(ma::filled(M, -1.0), *om, 1e-9, 1e-12, true));
  auto omm = npt::oracle(
      "A=np.ma.masked_array([10,20,30,40.],mask=[1,0,0,0]);"
      "B=np.ma.masked_array([1,2,3,4.],mask=[0,0,1,0]);"
      "a=np.ma.getmaskarray(A*B).astype(np.float64)");
  if (omm) CHECK(allclose(ma::getmask(M).astype(kFloat64), *omm, 0, 0, true));
}

TEST_CASE("ma divide vs numpy") {
  using namespace numpp; namespace ma = numpp::ma;
  ma::MaskedArray A = ma::masked_array(ma_vec({8, 9, 10, 12}), ma_mask({0, 0, 0, 1}));
  ma::MaskedArray B = ma::masked_array(ma_vec({2, 3, 0, 4}), ma_mask({0, 0, 1, 0}));
  ma::MaskedArray R = ma::divide(A, B);
  auto o = npt::oracle(
      "A=np.ma.masked_array([8,9,10,12.],mask=[0,0,0,1]);"
      "B=np.ma.masked_array([2,3,0,4.],mask=[0,0,1,0]);a=(A/B).filled(-1.)");
  if (o) CHECK(allclose(ma::filled(R, -1.0), *o, 1e-9, 1e-12, true));
  auto om = npt::oracle(
      "A=np.ma.masked_array([8,9,10,12.],mask=[0,0,0,1]);"
      "B=np.ma.masked_array([2,3,0,4.],mask=[0,0,1,0]);"
      "a=np.ma.getmaskarray(A/B).astype(np.float64)");
  if (om) CHECK(allclose(ma::getmask(R).astype(kFloat64), *om, 0, 0, true));
}

TEST_CASE("ma masked_less / masked_less_equal vs numpy") {
  using namespace numpp; namespace ma = numpp::ma;
  ndarray x = ma_vec({1, 2, 3, 4, 5});
  ma::MaskedArray L = ma::masked_less(x, 3.0);
  ma::MaskedArray LE = ma::masked_less_equal(x, 3.0);
  auto ol = npt::oracle("a=np.ma.masked_less([1,2,3,4,5.],3).filled(-1.)");
  if (ol) CHECK(allclose(ma::filled(L, -1.0), *ol, 1e-9, 1e-12, true));
  auto olm = npt::oracle("a=np.ma.getmaskarray(np.ma.masked_less([1,2,3,4,5.],3)).astype(np.float64)");
  if (olm) CHECK(allclose(ma::getmask(L).astype(kFloat64), *olm, 0, 0, true));
  auto ole = npt::oracle("a=np.ma.getmaskarray(np.ma.masked_less_equal([1,2,3,4,5.],3)).astype(np.float64)");
  if (ole) CHECK(allclose(ma::getmask(LE).astype(kFloat64), *ole, 0, 0, true));
}

TEST_CASE("ma masked_greater_equal / masked_not_equal vs numpy") {
  using namespace numpp; namespace ma = numpp::ma;
  ndarray x = ma_vec({1, 2, 3, 4, 5});
  ma::MaskedArray GE = ma::masked_greater_equal(x, 3.0);
  ma::MaskedArray NE = ma::masked_not_equal(x, 3.0);
  auto oge = npt::oracle("a=np.ma.getmaskarray(np.ma.masked_greater_equal([1,2,3,4,5.],3)).astype(np.float64)");
  if (oge) CHECK(allclose(ma::getmask(GE).astype(kFloat64), *oge, 0, 0, true));
  auto one = npt::oracle("a=np.ma.getmaskarray(np.ma.masked_not_equal([1,2,3,4,5.],3)).astype(np.float64)");
  if (one) CHECK(allclose(ma::getmask(NE).astype(kFloat64), *one, 0, 0, true));
  auto od = npt::oracle("a=np.ma.masked_not_equal([1,2,3,4,5.],3).filled(-1.)");
  if (od) CHECK(allclose(ma::filled(NE, -1.0), *od, 1e-9, 1e-12, true));
}

TEST_CASE("ma masked_inside / masked_outside vs numpy") {
  using namespace numpp; namespace ma = numpp::ma;
  ndarray x = ma_vec({0, 1, 2, 3, 4, 5});
  ma::MaskedArray IN = ma::masked_inside(x, 2.0, 4.0);
  ma::MaskedArray OUT = ma::masked_outside(x, 2.0, 4.0);
  auto oin = npt::oracle("a=np.ma.getmaskarray(np.ma.masked_inside([0,1,2,3,4,5.],2,4)).astype(np.float64)");
  if (oin) CHECK(allclose(ma::getmask(IN).astype(kFloat64), *oin, 0, 0, true));
  auto oout = npt::oracle("a=np.ma.getmaskarray(np.ma.masked_outside([0,1,2,3,4,5.],2,4)).astype(np.float64)");
  if (oout) CHECK(allclose(ma::getmask(OUT).astype(kFloat64), *oout, 0, 0, true));
  // swapped bounds behave identically
  ma::MaskedArray IN2 = ma::masked_inside(x, 4.0, 2.0);
  auto oin2 = npt::oracle("a=np.ma.getmaskarray(np.ma.masked_inside([0,1,2,3,4,5.],4,2)).astype(np.float64)");
  if (oin2) CHECK(allclose(ma::getmask(IN2).astype(kFloat64), *oin2, 0, 0, true));
}

TEST_CASE("ma masked_values vs numpy") {
  using namespace numpp; namespace ma = numpp::ma;
  ndarray x = ma_vec({1.0, 1.1, 2.0, 1.0});
  ma::MaskedArray V = ma::masked_values(x, 1.0);
  auto ov = npt::oracle("a=np.ma.masked_values([1.,1.1,2.,1.0],1.0).filled(-1.)");
  if (ov) CHECK(allclose(ma::filled(V, -1.0), *ov, 1e-9, 1e-12, true));
  auto ovm = npt::oracle("a=np.ma.getmaskarray(np.ma.masked_values([1.,1.1,2.,1.0],1.0)).astype(np.float64)");
  if (ovm) CHECK(allclose(ma::getmask(V).astype(kFloat64), *ovm, 0, 0, true));
}

TEST_CASE("ma getdata returns underlying data") {
  using namespace numpp; namespace ma = numpp::ma;
  ndarray x = ma_vec({3, 6, 9});
  ma::MaskedArray M = ma::masked_array(x, ma_mask({0, 1, 0}));
  auto o = npt::oracle("a=np.ma.getdata(np.ma.masked_array([3,6,9.],mask=[0,1,0])).astype(np.float64)");
  if (o) CHECK(allclose(ma::getdata(M).astype(kFloat64), *o, 1e-9, 1e-12, true));
}
