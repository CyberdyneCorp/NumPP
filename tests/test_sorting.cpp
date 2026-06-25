#include <limits>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s\n", label); return; }
  CHECK(allclose(got, *o, 1e-9, 1e-12, true));
}
static ndarray vec(const std::vector<double>& v, DType dt = kFloat64) {
  ndarray a({(int64_t)v.size()}, kFloat64);
  for (size_t i = 0; i < v.size(); ++i) a.set_item<double>({(int64_t)i}, v[i]);
  return a.astype(dt);
}

TEST_CASE("sort 1d vs numpy") {
  chk(sort(vec({3, 1, 2, 5, 4})), "a=np.sort(np.array([3.,1,2,5,4]))", "sort1d");
  chk(sort(vec({3, 1, 2, 5, 4}, kInt64)), "a=np.sort(np.array([3,1,2,5,4]))", "sort1d int");
}

TEST_CASE("sort puts NaN at the end like numpy") {
  ndarray a = vec({3.0, std::nan(""), 1.0, 2.0});
  ndarray s = sort(a);
  CHECK(s.item<double>({0}) == 1.0);
  CHECK(s.item<double>({1}) == 2.0);
  CHECK(s.item<double>({2}) == 3.0);
  CHECK(std::isnan(s.item<double>({3})));
}

TEST_CASE("sort along axis") {
  ndarray a = arange(0.0, 6.0, 1.0).reshape({2, 3});  // already sorted rows
  ndarray r = a.index({IndexItem{Slice{std::nullopt, std::nullopt, -1}}, IndexItem{Slice{std::nullopt, std::nullopt, -1}}}).copy();
  chk(sort(r, int64_t{0}), "x=np.arange(6.).reshape(2,3)[::-1,::-1].copy();a=np.sort(x,axis=0)", "sort axis0");
  chk(sort(r, int64_t{1}), "x=np.arange(6.).reshape(2,3)[::-1,::-1].copy();a=np.sort(x,axis=1)", "sort axis1");
}

TEST_CASE("sort axis=None flattens") {
  ndarray a = arange(0.0, 6.0, 1.0).reshape({2, 3});
  ndarray r = a.transpose().copy();
  chk(sort(r, std::nullopt), "a=np.sort(np.arange(6.).reshape(2,3).T.copy(),axis=None)", "sort none");
}

TEST_CASE("argsort: a[argsort]==sort") {
  ndarray a = vec({3, 1, 4, 1, 5, 9, 2, 6});
  ndarray idx = argsort(a);
  ndarray s = sort(a);
  for (int64_t i = 0; i < a.size(); ++i)
    CHECK(a.item<double>({idx.item<int64_t>({i})}) == s.item<double>({i}));
  CHECK(idx.dtype() == kInt64);
}

TEST_CASE("argsort stable keeps tie order") {
  // values with ties; stable argsort must keep original order of equal keys
  ndarray a = vec({2, 1, 2, 1, 2});
  ndarray idx = argsort(a, int64_t{-1}, SortKind::Stable);
  // the two 1s are at positions 1,3 -> should come first in order [1,3], then 2s [0,2,4]
  int64_t want[5] = {1, 3, 0, 2, 4};
  for (int i = 0; i < 5; ++i) CHECK(idx.item<int64_t>({i}) == want[i]);
}

TEST_CASE("argsort along axis vs numpy reconstruction") {
  ndarray a = vec({5, 2, 8, 1, 9, 3}).reshape({2, 3});
  ndarray idx = argsort(a, int64_t{1});
  auto o = npt::oracle("x=np.array([5.,2,8,1,9,3]).reshape(2,3);a=np.take_along_axis(x,np.argsort(x,axis=1),axis=1)");
  ndarray s = sort(a, int64_t{1});
  if (o) CHECK(allclose(s, *o, 1e-9, 1e-12));
}
