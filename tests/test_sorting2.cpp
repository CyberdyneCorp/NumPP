#include <limits>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
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

TEST_CASE("searchsorted left/right vs numpy") {
  ndarray a = vec({1, 2, 3, 4, 5});
  ndarray v = vec({0, 2, 2, 6});
  chk(searchsorted(a, v, "left"), "a=np.searchsorted(np.arange(1,6.),np.array([0.,2,2,6]),'left')", "ss left");
  chk(searchsorted(a, v, "right"), "a=np.searchsorted(np.arange(1,6.),np.array([0.,2,2,6]),'right')", "ss right");
}

TEST_CASE("partition: kth in place, partition property") {
  ndarray a = vec({7, 2, 9, 1, 5, 3, 8, 4});
  int64_t k = 3;
  ndarray p = partition(a, k);
  ndarray s = sort(a);
  CHECK(p.item<double>({k}) == s.item<double>({k}));
  for (int64_t i = 0; i < k; ++i) CHECK(p.item<double>({i}) <= p.item<double>({k}));
  for (int64_t i = k + 1; i < a.size(); ++i) CHECK(p.item<double>({i}) >= p.item<double>({k}));
  // argpartition: a[argpartition][k] == partition[k]
  ndarray ip = argpartition(a, k);
  CHECK(a.item<double>({ip.item<int64_t>({k})}) == p.item<double>({k}));
}

TEST_CASE("argmin/argmax vs numpy (incl axis and NaN)") {
  chk(argmin(vec({3, 1, 4, 1, 5})), "a=np.array(np.argmin(np.array([3.,1,4,1,5])))", "argmin");
  chk(argmax(vec({3, 1, 4, 1, 5})), "a=np.array(np.argmax(np.array([3.,1,4,1,5])))", "argmax");
  ndarray m = vec({5, 2, 8, 1, 9, 3}).reshape({2, 3});
  chk(argmin(m, int64_t{1}), "a=np.argmin(np.array([5.,2,8,1,9,3]).reshape(2,3),axis=1)", "argmin axis1");
  chk(argmax(m, int64_t{0}), "a=np.argmax(np.array([5.,2,8,1,9,3]).reshape(2,3),axis=0)", "argmax axis0");
  // NaN: numpy argmin returns the NaN index
  ndarray n = vec({3, std::nan(""), 1});
  CHECK(argmin(n).item<int64_t>({}) == 1);
  CHECK(argmax(n).item<int64_t>({}) == 1);
}

TEST_CASE("flatnonzero and count_nonzero vs numpy") {
  ndarray a = vec({0, 3, 0, 0, 7, 1, 0}).reshape({7});
  chk(flatnonzero(a), "a=np.flatnonzero(np.array([0.,3,0,0,7,1,0]))", "flatnonzero");
  ndarray m = vec({0, 1, 0, 2, 0, 0}).reshape({2, 3});
  CHECK(count_nonzero(m).item<int64_t>({}) == 2);
  chk(count_nonzero(m, int64_t{1}), "a=np.count_nonzero(np.array([0.,1,0,2,0,0]).reshape(2,3),axis=1)", "count axis1");
}
