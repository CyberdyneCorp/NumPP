#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace numpp;
namespace ma = numpp::ma;

// Build a bool mask array from a predicate over arange(0,n).
static ndarray mask_gt(int64_t n, double thr) {
  ndarray m(Shape{n}, kBool);
  for (int64_t i = 0; i < n; ++i) m.set_item<bool>({i}, static_cast<double>(i) > thr);
  return m;
}

TEST_CASE("ma sum masked_where vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto m = ma::masked_where(mask_gt(6, 3.0), x);
  auto o = npt::oracle("a=np.ma.masked_where(np.arange(6.)>3, np.arange(6.)).sum()");
  if (o) CHECK(allclose(ma::sum(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma mean masked_where vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto m = ma::masked_where(mask_gt(6, 3.0), x);
  auto o = npt::oracle("a=np.ma.masked_where(np.arange(6.)>3, np.arange(6.)).mean()");
  if (o) CHECK(allclose(ma::mean(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma min masked_where vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto m = ma::masked_where(mask_gt(6, 3.0), x);
  auto o = npt::oracle("a=np.ma.masked_where(np.arange(6.)>3, np.arange(6.)).min()");
  if (o) CHECK(allclose(ma::min(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma max masked_where vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto m = ma::masked_where(mask_gt(6, 3.0), x);
  auto o = npt::oracle("a=np.ma.masked_where(np.arange(6.)>3, np.arange(6.)).max()");
  if (o) CHECK(allclose(ma::max(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma filled masked_where vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto m = ma::masked_where(mask_gt(6, 3.0), x);
  auto o = npt::oracle("a=np.ma.masked_where(np.arange(6.)>3, np.arange(6.)).filled(-1.)");
  if (o) CHECK(allclose(ma::filled(m, -1.0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma compressed masked_where vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto m = ma::masked_where(mask_gt(6, 3.0), x);
  auto o = npt::oracle("a=np.ma.masked_where(np.arange(6.)>3, np.arange(6.)).compressed()");
  if (o) CHECK(allclose(ma::compressed(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma count masked_equal") {
  ndarray x = arange(0., 4., 1., kFloat64);
  x.set_item<double>({1}, 2.0);  // [0,2,2,3]
  auto m = ma::masked_equal(x, 2.0);
  CHECK(ma::count(m) == 2);
}

TEST_CASE("ma masked_equal sum vs numpy") {
  ndarray x(Shape{4}, kFloat64);
  x.set_item<double>({0}, 1.0); x.set_item<double>({1}, 2.0);
  x.set_item<double>({2}, 2.0); x.set_item<double>({3}, 3.0);
  auto m = ma::masked_equal(x, 2.0);
  auto o = npt::oracle("a=np.ma.masked_equal([1.,2.,2.,3.],2).sum()");
  if (o) CHECK(allclose(ma::sum(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma masked_greater mean vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto m = ma::masked_greater(x, 2.0);
  auto o = npt::oracle("a=np.ma.masked_greater(np.arange(6.),2.).mean()");
  if (o) CHECK(allclose(ma::mean(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma masked_greater compressed vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto m = ma::masked_greater(x, 2.0);
  auto o = npt::oracle("a=np.ma.masked_greater(np.arange(6.),2.).compressed()");
  if (o) CHECK(allclose(ma::compressed(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma masked_invalid mean vs numpy") {
  ndarray x(Shape{4}, kFloat64);
  x.set_item<double>({0}, 1.0);
  x.set_item<double>({1}, std::nan(""));
  x.set_item<double>({2}, 3.0);
  x.set_item<double>({3}, std::numeric_limits<double>::infinity());
  auto m = ma::masked_invalid(x);
  auto o = npt::oracle("a=np.ma.masked_invalid([1.,np.nan,3.,np.inf]).mean()");
  if (o) CHECK(allclose(ma::mean(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma masked_invalid count") {
  ndarray x(Shape{4}, kFloat64);
  x.set_item<double>({0}, 1.0);
  x.set_item<double>({1}, std::nan(""));
  x.set_item<double>({2}, 3.0);
  x.set_item<double>({3}, std::numeric_limits<double>::infinity());
  auto m = ma::masked_invalid(x);
  CHECK(ma::count(m) == 2);
}

TEST_CASE("ma masked_array explicit mask sum vs numpy") {
  ndarray x = arange(0., 5., 1., kFloat64);
  ndarray mk(Shape{5}, kBool);
  for (int64_t i = 0; i < 5; ++i) mk.set_item<bool>({i}, i % 2 == 0);
  auto m = ma::masked_array(x, mk);
  auto o = npt::oracle("a=np.ma.array(np.arange(5.),mask=[1,0,1,0,1]).sum()");
  if (o) CHECK(allclose(ma::sum(m), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ma masked_array max vs numpy") {
  ndarray x = arange(0., 5., 1., kFloat64);
  ndarray mk(Shape{5}, kBool);
  for (int64_t i = 0; i < 5; ++i) mk.set_item<bool>({i}, i % 2 == 0);
  auto m = ma::masked_array(x, mk);
  auto o = npt::oracle("a=np.ma.array(np.arange(5.),mask=[1,0,1,0,1]).max()");
  if (o) CHECK(allclose(ma::max(m), *o, 1e-9, 1e-12, true));
}
