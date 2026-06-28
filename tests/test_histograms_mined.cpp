// Cases mined from numpy's numpy/lib/tests/test_histograms.py: histogram counts
// and bin edges (auto and explicit range), histogram_bin_edges, histogram2d.
// (NumPP's histogram exposes bins+range; density=/weights= are not part of the
// numpp.histogram surface, so those numpy paths are out of scope here.)
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <string>
#include <utility>
#include <vector>

using namespace numpp;

namespace {
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined histograms: counts and bin edges (auto range) vs numpy") {
  ndarray a = dval({1, 2, 1, 3, 4, 1});
  Histogram h = histogram(a, 4);
  auto oh = npt::oracle("a=np.histogram([1,2,1,3,4,1.],bins=4)[0]");
  if (oh) CHECK(allclose(h.hist.astype(kFloat64), *oh, 0, 0, true));
  auto oe = npt::oracle("a=np.histogram([1,2,1,3,4,1.],bins=4)[1]");
  if (oe) CHECK(allclose(h.bin_edges, *oe, 1e-12, 1e-12, true));
}

TEST_CASE("mined histograms: counts with explicit range vs numpy") {
  ndarray a = dval({1, 2, 1, 3, 4, 1});
  Histogram h = histogram(a, 5, std::make_pair(0.0, 5.0));
  auto oh = npt::oracle("a=np.histogram([1,2,1,3,4,1.],bins=5,range=(0,5))[0]");
  if (oh) CHECK(allclose(h.hist.astype(kFloat64), *oh, 0, 0, true));
  auto oe = npt::oracle("a=np.histogram([1,2,1,3,4,1.],bins=5,range=(0,5))[1]");
  if (oe) CHECK(allclose(h.bin_edges, *oe, 1e-12, 1e-12, true));
}

TEST_CASE("mined histograms: histogram_bin_edges vs numpy") {
  ndarray a = dval({1, 2, 3, 4});
  auto o = npt::oracle("a=np.histogram_bin_edges([1,2,3,4.],bins=3)");
  if (o) CHECK(allclose(histogram_bin_edges(a, 3), *o, 1e-12, 1e-12, true));
  auto o2 = npt::oracle("a=np.histogram_bin_edges([1,2,3,4.],bins=2,range=(0,4))");
  if (o2) CHECK(allclose(histogram_bin_edges(a, 2, std::make_pair(0.0, 4.0)), *o2, 1e-12, 1e-12, true));
}

TEST_CASE("mined histograms: histogram2d counts and edges vs numpy") {
  ndarray x = dval({1, 2, 3, 4}), y = dval({1, 2, 3, 4});
  Histogram2DResult r = histogram2d(x, y, 2);
  auto oh = npt::oracle("a=np.histogram2d([1,2,3,4.],[1,2,3,4.],bins=2)[0].ravel()");
  if (oh) CHECK(allclose(r.H.astype(kFloat64).ravel(), *oh, 0, 0, true));
  auto ox = npt::oracle("a=np.histogram2d([1,2,3,4.],[1,2,3,4.],bins=2)[1]");
  if (ox) CHECK(allclose(r.xedges, *ox, 1e-12, 1e-12, true));
  auto oy = npt::oracle("a=np.histogram2d([1,2,3,4.],[1,2,3,4.],bins=2)[2]");
  if (oy) CHECK(allclose(r.yedges, *oy, 1e-12, 1e-12, true));
}
