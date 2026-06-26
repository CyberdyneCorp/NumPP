#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("histogram2d counts vs numpy") {
  ndarray x = arange(0., 8., 1., kFloat64);
  ndarray y = arange(7., -1., -1., kFloat64);  // [7,6,5,4,3,2,1,0]
  auto H = npt::oracle("a=np.histogram2d(np.arange(8.0), np.arange(7,-1,-1,dtype=float), bins=4)[0]");
  if (H) CHECK(allclose(histogram2d(x, y, 4).H, *H, 1e-9, 1e-12, true));
}

TEST_CASE("histogram2d edges vs numpy") {
  ndarray x = arange(0., 8., 1., kFloat64);
  ndarray y = arange(7., -1., -1., kFloat64);
  auto xe = npt::oracle("a=np.histogram2d(np.arange(8.0), np.arange(7,-1,-1,dtype=float), bins=4)[1]");
  auto ye = npt::oracle("a=np.histogram2d(np.arange(8.0), np.arange(7,-1,-1,dtype=float), bins=4)[2]");
  Histogram2DResult r = histogram2d(x, y, 4);
  if (xe) CHECK(allclose(r.xedges, *xe, 1e-9, 1e-12, true));
  if (ye) CHECK(allclose(r.yedges, *ye, 1e-9, 1e-12, true));
}

TEST_CASE("histogram2d default bins vs numpy") {
  ndarray x = arange(0., 12., 1., kFloat64);
  ndarray y = arange(0., 24., 2., kFloat64);  // [0,2,...,22]
  auto H = npt::oracle("a=np.histogram2d(np.arange(12.0), np.arange(0,24,2,dtype=float))[0]");
  if (H) CHECK(allclose(histogram2d(x, y).H, *H, 1e-9, 1e-12, true));
}

TEST_CASE("histogramdd counts vs numpy") {
  // sample (6, 2)
  ndarray s = arange(0., 12., 1., kFloat64).reshape({6, 2});
  auto H = npt::oracle("a=np.histogramdd(np.arange(12.0).reshape(6,2), bins=3)[0]");
  if (H) CHECK(allclose(histogramdd(s, 3).H, *H, 1e-9, 1e-12, true));
}

TEST_CASE("histogramdd 3d counts vs numpy") {
  ndarray s = arange(0., 12., 1., kFloat64).reshape({4, 3});
  HistogramDDResult r = histogramdd(s, 2);
  auto H = npt::oracle("a=np.histogramdd(np.arange(12.0).reshape(4,3), bins=2)[0]");
  if (H) CHECK(allclose(r.H, *H, 1e-9, 1e-12, true));
  CHECK(r.edges.size() == 3);
  auto e0 = npt::oracle("a=np.histogramdd(np.arange(12.0).reshape(4,3), bins=2)[1][0]");
  if (e0) CHECK(allclose(r.edges[0], *e0, 1e-9, 1e-12, true));
}

TEST_CASE("nanquantile flatten vs numpy") {
  ndarray a = arange(0., 6., 1., kFloat64);
  a.set_item<double>({2}, std::numeric_limits<double>::quiet_NaN());
  auto o = npt::oracle("a=np.nanquantile(np.array([0.,1.,np.nan,3.,4.,5.]), 0.4)");
  if (o) CHECK(allclose(nanquantile(a, 0.4), *o, 1e-9, 1e-12, true));
  auto o2 = npt::oracle("a=np.nanquantile(np.array([0.,1.,np.nan,3.,4.,5.]), 0.0)");
  if (o2) CHECK(allclose(nanquantile(a, 0.0), *o2, 1e-9, 1e-12, true));
  auto o3 = npt::oracle("a=np.nanquantile(np.array([0.,1.,np.nan,3.,4.,5.]), 1.0)");
  if (o3) CHECK(allclose(nanquantile(a, 1.0), *o3, 1e-9, 1e-12, true));
}

TEST_CASE("nanquantile along axis vs numpy") {
  ndarray a = arange(0., 12., 1., kFloat64).reshape({3, 4});
  a.set_item<double>({1, 2}, std::numeric_limits<double>::quiet_NaN());
  auto o = npt::oracle(
    "m=np.arange(12.0).reshape(3,4); m[1,2]=np.nan; a=np.nanquantile(m, 0.5, axis=1)");
  if (o) CHECK(allclose(nanquantile(a, 0.5, 1), *o, 1e-9, 1e-12, true));
  auto o0 = npt::oracle(
    "m=np.arange(12.0).reshape(3,4); m[1,2]=np.nan; a=np.nanquantile(m, 0.25, axis=0)");
  if (o0) CHECK(allclose(nanquantile(a, 0.25, 0), *o0, 1e-9, 1e-12, true));
}

TEST_CASE("cov_weighted vs numpy") {
  ndarray m = arange(0., 8., 1., kFloat64).reshape({2, 4});
  ndarray w = arange(1., 5., 1., kFloat64);  // [1,2,3,4]
  auto o = npt::oracle(
    "a=np.cov(np.arange(8.0).reshape(2,4), aweights=np.array([1.,2.,3.,4.]))");
  if (o) CHECK(allclose(cov_weighted(m, w), *o, 1e-9, 1e-12, true));
}

TEST_CASE("cov_weighted ddof=0 vs numpy") {
  ndarray m = arange(0., 8., 1., kFloat64).reshape({2, 4});
  ndarray w = arange(1., 5., 1., kFloat64);
  auto o = npt::oracle(
    "a=np.cov(np.arange(8.0).reshape(2,4), aweights=np.array([1.,2.,3.,4.]), ddof=0)");
  if (o) CHECK(allclose(cov_weighted(m, w, true, 0), *o, 1e-9, 1e-12, true));
}

TEST_CASE("cov_weighted rowvar=false vs numpy") {
  // observations along rows: (4 obs, 2 vars)
  ndarray m = arange(0., 8., 1., kFloat64).reshape({4, 2});
  ndarray w = arange(1., 5., 1., kFloat64);
  auto o = npt::oracle(
    "a=np.cov(np.arange(8.0).reshape(4,2), rowvar=False, aweights=np.array([1.,2.,3.,4.]))");
  if (o) CHECK(allclose(cov_weighted(m, w, false), *o, 1e-9, 1e-12, true));
}
