#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

// ============================= cumulative =============================
TEST_CASE("cumsum flat vs numpy") {
  ndarray x = arange(1., 7., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.cumsum(np.arange(1,7.).reshape(2,3))");
  if (o) CHECK(allclose(cumsum(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("cumsum axis0 vs numpy") {
  ndarray x = arange(1., 7., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.cumsum(np.arange(1,7.).reshape(2,3),axis=0)");
  if (o) CHECK(allclose(cumsum(x, 0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("cumsum axis1 vs numpy") {
  ndarray x = arange(1., 7., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.cumsum(np.arange(1,7.).reshape(2,3),axis=1)");
  if (o) CHECK(allclose(cumsum(x, 1), *o, 1e-9, 1e-12, true));
}
TEST_CASE("cumsum int keeps int dtype") {
  ndarray x = arange(1, 6, 1, kInt64);
  auto o = npt::oracle("a=np.cumsum(np.arange(1,6))");
  if (o) {
    ndarray r = cumsum(x);
    CHECK(r.dtype() == kInt64);
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}
TEST_CASE("cumprod axis1 vs numpy") {
  ndarray x = arange(1., 7., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.cumprod(np.arange(1,7.).reshape(2,3),axis=1)");
  if (o) CHECK(allclose(cumprod(x, 1), *o, 1e-9, 1e-12, true));
}
TEST_CASE("nancumsum drops nan vs numpy") {
  ndarray x = arange(1., 6., 1., kFloat64);
  x.set_item<double>({2}, std::nan(""));
  auto o = npt::oracle("a=np.nancumsum([1.,2.,np.nan,4.,5.])");
  if (o) CHECK(allclose(nancumsum(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("nancumprod drops nan vs numpy") {
  ndarray x = arange(1., 6., 1., kFloat64);
  x.set_item<double>({2}, std::nan(""));
  auto o = npt::oracle("a=np.nancumprod([1.,2.,np.nan,4.,5.])");
  if (o) CHECK(allclose(nancumprod(x), *o, 1e-9, 1e-12, true));
}

// ============================= differences =============================
TEST_CASE("diff n1 vs numpy") {
  ndarray x = arange(0., 10., 1., kFloat64).reshape({2, 5});
  auto o = npt::oracle("a=np.diff(np.arange(10.).reshape(2,5))");
  if (o) CHECK(allclose(diff(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("diff n2 axis0 vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({4, 3});
  auto o = npt::oracle("a=np.diff(np.arange(12.).reshape(4,3),n=2,axis=0)");
  if (o) CHECK(allclose(diff(x, 2, 0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("ediff1d vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.ediff1d(np.arange(6.).reshape(2,3))");
  if (o) CHECK(allclose(ediff1d(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("gradient 1d vs numpy") {
  ndarray x = arange(0., 5., 1., kFloat64);
  x.set_item<double>({0}, 1.0); x.set_item<double>({1}, 2.0); x.set_item<double>({2}, 4.0);
  x.set_item<double>({3}, 7.0); x.set_item<double>({4}, 11.0);
  auto o = npt::oracle("a=np.gradient(np.array([1.,2.,4.,7.,11.]))");
  if (o) CHECK(allclose(gradient(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("ptp flat vs numpy") {
  ndarray x = arange(3., 9., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.ptp(np.arange(3,9.).reshape(2,3))");
  if (o) CHECK(allclose(ptp(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("ptp axis1 vs numpy") {
  ndarray x = arange(3., 9., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.ptp(np.arange(3,9.).reshape(2,3),axis=1)");
  if (o) CHECK(allclose(ptp(x, 1), *o, 1e-9, 1e-12, true));
}

// ============================= order statistics =============================
TEST_CASE("median flat odd vs numpy") {
  ndarray x = arange(1., 8., 1., kFloat64);
  auto o = npt::oracle("a=np.median(np.arange(1,8.))");
  if (o) CHECK(allclose(median(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("median flat even vs numpy") {
  ndarray x = arange(1., 9., 1., kFloat64);
  auto o = npt::oracle("a=np.median(np.arange(1,9.))");
  if (o) CHECK(allclose(median(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("median axis0 vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64).reshape({4, 3});
  auto o = npt::oracle("a=np.median(np.arange(12.).reshape(4,3),axis=0)");
  if (o) CHECK(allclose(median(x, 0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("percentile 25 vs numpy") {
  ndarray x = arange(0., 11., 1., kFloat64);
  auto o = npt::oracle("a=np.percentile(np.arange(11.),25)");
  if (o) CHECK(allclose(percentile(x, 25.0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("percentile 90 axis1 vs numpy") {
  ndarray x = arange(0., 20., 1., kFloat64).reshape({4, 5});
  auto o = npt::oracle("a=np.percentile(np.arange(20.).reshape(4,5),90,axis=1)");
  if (o) CHECK(allclose(percentile(x, 90.0, 1), *o, 1e-9, 1e-12, true));
}
TEST_CASE("quantile 0.3 vs numpy") {
  ndarray x = arange(0., 11., 1., kFloat64);
  auto o = npt::oracle("a=np.quantile(np.arange(11.),0.3)");
  if (o) CHECK(allclose(quantile(x, 0.3), *o, 1e-9, 1e-12, true));
}
TEST_CASE("nanmedian vs numpy") {
  ndarray x = arange(1., 8., 1., kFloat64);
  x.set_item<double>({3}, std::nan(""));
  auto o = npt::oracle("a=np.nanmedian([1.,2.,3.,np.nan,5.,6.,7.])");
  if (o) CHECK(allclose(nanmedian(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("nanpercentile 40 vs numpy") {
  ndarray x = arange(1., 8., 1., kFloat64);
  x.set_item<double>({3}, std::nan(""));
  auto o = npt::oracle("a=np.nanpercentile([1.,2.,3.,np.nan,5.,6.,7.],40)");
  if (o) CHECK(allclose(nanpercentile(x, 40.0), *o, 1e-9, 1e-12, true));
}

// ============================= correlation / misc =============================
TEST_CASE("average unweighted equals mean") {
  ndarray x = arange(0., 6., 1., kFloat64);
  auto o = npt::oracle("a=np.average(np.arange(6.))");
  if (o) CHECK(allclose(average(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("average weighted flat vs numpy") {
  ndarray x = arange(1., 5., 1., kFloat64);
  ndarray w = arange(1., 5., 1., kFloat64);
  auto o = npt::oracle("a=np.average(np.arange(1,5.),weights=np.arange(1,5.))");
  if (o) CHECK(allclose(average(x, std::nullopt, &w), *o, 1e-9, 1e-12, true));
}
TEST_CASE("average weighted axis1 vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray w = arange(1., 4., 1., kFloat64);
  auto o = npt::oracle("a=np.average(np.arange(6.).reshape(2,3),axis=1,weights=np.array([1.,2.,3.]))");
  if (o) CHECK(allclose(average(x, 1, &w), *o, 1e-9, 1e-12, true));
}
TEST_CASE("cov rowvar vs numpy") {
  ndarray m = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.cov(np.arange(6.).reshape(2,3))");
  if (o) CHECK(allclose(cov(m), *o, 1e-9, 1e-12, true));
}
TEST_CASE("cov ddof0 vs numpy") {
  ndarray m = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.cov(np.arange(6.).reshape(2,3),ddof=0)");
  if (o) CHECK(allclose(cov(m, true, 0), *o, 1e-9, 1e-12, true));
}
TEST_CASE("corrcoef vs numpy") {
  ndarray m = arange(0., 6., 1., kFloat64).reshape({2, 3});
  m.set_item<double>({1, 0}, 0.0); m.set_item<double>({1, 1}, 2.0); m.set_item<double>({1, 2}, 1.0);
  auto o = npt::oracle("a=np.corrcoef(np.array([[0.,1.,2.],[0.,2.,1.]]))");
  if (o) CHECK(allclose(corrcoef(m), *o, 1e-9, 1e-12, true));
}
TEST_CASE("digitize right=false vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  ndarray bins = arange(1., 5., 1., kFloat64);
  auto o = npt::oracle("a=np.digitize(np.arange(6.),np.array([1.,2.,3.,4.]))");
  if (o) CHECK(allclose(digitize(x, bins), *o, 1e-9, 1e-12, true));
}
TEST_CASE("digitize right=true vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  ndarray bins = arange(1., 5., 1., kFloat64);
  auto o = npt::oracle("a=np.digitize(np.arange(6.),np.array([1.,2.,3.,4.]),right=True)");
  if (o) CHECK(allclose(digitize(x, bins, true), *o, 1e-9, 1e-12, true));
}
TEST_CASE("nanargmin vs numpy") {
  ndarray x = arange(1., 6., 1., kFloat64);
  x.set_item<double>({0}, std::nan(""));
  auto o = npt::oracle("a=np.nanargmin([np.nan,2.,3.,4.,5.])");
  if (o) CHECK(allclose(nanargmin(x), *o, 1e-9, 1e-12, true));
}
TEST_CASE("nanargmax vs numpy") {
  ndarray x = arange(1., 6., 1., kFloat64);
  x.set_item<double>({4}, std::nan(""));
  auto o = npt::oracle("a=np.nanargmax([1.,2.,3.,4.,np.nan])");
  if (o) CHECK(allclose(nanargmax(x), *o, 1e-9, 1e-12, true));
}
