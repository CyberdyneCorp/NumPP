#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chkf(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s\n", label); return; }
  CHECK(allclose(got, *o, 1e-9, 1e-12, true));
}
static void chki(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s\n", label); return; }
  ndarray g = got.reshape({got.size()}), oo = o->reshape({o->size()});
  CHECK(g.size() == oo.size());
  for (int64_t i = 0; i < g.size(); ++i) CHECK(g.item<int64_t>({i}) == oo.item<int64_t>({i}));
}
static ndarray vec(const std::vector<double>& v, DType dt = kFloat64) {
  ndarray a({(int64_t)v.size()}, kFloat64);
  for (size_t i = 0; i < v.size(); ++i) a.set_item<double>({(int64_t)i}, v[i]);
  return a.astype(dt);
}

TEST_CASE("unique values/index/inverse/counts vs numpy") {
  ndarray a = vec({3, 1, 2, 1, 3, 3, 2}, kInt64);
  chkf(unique(a), "a=np.unique(np.array([3,1,2,1,3,3,2]))", "unique vals");
  auto u = unique_ex(a, true, true, true);
  chki(u.index, "a=np.unique(np.array([3,1,2,1,3,3,2]),return_index=True)[1]", "unique index");
  chki(u.inverse, "a=np.unique(np.array([3,1,2,1,3,3,2]),return_inverse=True)[1].ravel()", "unique inverse");
  chki(u.counts, "a=np.unique(np.array([3,1,2,1,3,3,2]),return_counts=True)[1]", "unique counts");
  // reconstruct a from unique[inverse]
  for (int64_t i = 0; i < a.size(); ++i)
    CHECK(u.values.item<int64_t>({u.inverse.item<int64_t>({i})}) == a.item<int64_t>({i}));
}

TEST_CASE("set operations vs numpy") {
  ndarray a = vec({1, 2, 3, 4}, kInt64), b = vec({3, 4, 5, 6}, kInt64);
  chki(in1d(a, b).astype(kInt64), "a=np.in1d(np.array([1,2,3,4]),np.array([3,4,5,6])).astype(np.int64)", "in1d");
  chki(intersect1d(a, b), "a=np.intersect1d(np.array([1,2,3,4]),np.array([3,4,5,6]))", "intersect1d");
  chki(union1d(a, b), "a=np.union1d(np.array([1,2,3,4]),np.array([3,4,5,6]))", "union1d");
  chki(setdiff1d(a, b), "a=np.setdiff1d(np.array([1,2,3,4]),np.array([3,4,5,6]))", "setdiff1d");
  CHECK(isin(a, b).shape() == a.shape());
}

TEST_CASE("bincount vs numpy") {
  ndarray x = vec({0, 1, 1, 2, 2, 2, 4}, kInt64);
  chki(bincount(x), "a=np.bincount(np.array([0,1,1,2,2,2,4]))", "bincount");
  chki(bincount(x, nullptr, 8), "a=np.bincount(np.array([0,1,1,2,2,2,4]),minlength=8)", "bincount minlen");
  ndarray w = vec({0.5, 1, 1, 2, 2, 2, 3});
  chkf(bincount(x, &w), "a=np.bincount(np.array([0,1,1,2,2,2,4]),weights=np.array([0.5,1,1,2,2,2,3]))", "bincount weights");
}

TEST_CASE("histogram vs numpy") {
  ndarray a = vec({0.1, 0.5, 0.9, 1.5, 2.5, 2.9, 1.1, 0.3});
  auto h = histogram(a, 4);
  chki(h.hist, "a=np.histogram(np.array([0.1,0.5,0.9,1.5,2.5,2.9,1.1,0.3]),bins=4)[0]", "hist counts");
  chkf(h.bin_edges, "a=np.histogram(np.array([0.1,0.5,0.9,1.5,2.5,2.9,1.1,0.3]),bins=4)[1]", "hist edges");
  auto h2 = histogram(a, 5, std::make_pair(0.0, 3.0));
  chki(h2.hist, "a=np.histogram(np.array([0.1,0.5,0.9,1.5,2.5,2.9,1.1,0.3]),bins=5,range=(0,3))[0]", "hist range");
}
