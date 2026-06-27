// Deeper test_multiarray.py mining: fancy/boolean indexing (negative/repeated,
// 2-D masks), take negative, put/place, choose, compress, ravel/unravel_index,
// and 0-d arrays — vs the live oracle.
#include "numpp/numpp.hpp"
#include "numpp/indexing/indexing.hpp"
#include "numpp/indexing/indexing_extra.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <vector>

using namespace numpp;

namespace {
ndarray i64(const std::vector<int64_t>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int64_t>()[i] = v[i];
  return a;
}
ndarray bvec(const std::vector<bool>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kBool, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<bool>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined multiarray: fancy index 1-D with negative/repeated vs numpy") {
  ndarray a = arange(10.0, 20.0, 1.0);
  ndarray idx = i64({0, 2, -1, 1, 1, -3});
  auto o = npt::oracle("a=np.arange(10,20.)[[0,2,-1,1,1,-3]]");
  if (o) CHECK(allclose(fancy_index(a, idx), *o, 0, 0, true));
}

TEST_CASE("mined multiarray: fancy_index is flat (a.flat[idx]) on a 2-D array vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4});
  ndarray idx = i64({2, 0, 1, -1});
  auto o = npt::oracle("A=np.arange(12.).reshape(3,4); a=A.flat[[2,0,1,-1]]");
  if (o) CHECK(allclose(fancy_index(A, idx), *o, 0, 0, true));
}

TEST_CASE("mined multiarray: take(axis=0) selects rows of a 2-D array vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4});
  ndarray idx = i64({2, 0, 1, -1});
  auto o = npt::oracle("A=np.arange(12.).reshape(3,4); a=A[[2,0,1,-1]]");  // axis-0 advanced index
  if (o) CHECK(allclose(take(A, idx, int64_t{0}), *o, 0, 0, true));
}

TEST_CASE("mined multiarray: boolean index (1-D and 2-D mask) vs numpy") {
  ndarray a = arange(0.0, 6.0, 1.0);
  ndarray m = bvec({true, false, true, true, false, true});
  auto o1 = npt::oracle("a=np.arange(6.)[np.array([True,False,True,True,False,True])]");
  if (o1) CHECK(allclose(boolean_index(a, m), *o1, 0, 0, true));
  ndarray A = arange(0.0, 6.0, 1.0).reshape({2, 3});
  ndarray M = bvec({true, false, true, false, true, true}).reshape({2, 3});
  auto o2 = npt::oracle(
      "A=np.arange(6.).reshape(2,3); a=A[np.array([[True,False,True],[False,True,True]])]");
  if (o2) CHECK(allclose(boolean_index(A, M), *o2, 0, 0, true));
}

TEST_CASE("mined multiarray: take with negative indices vs numpy") {
  ndarray a = arange(0.0, 10.0, 1.0);
  ndarray idx = i64({-1, -2, 0, 3});
  auto o = npt::oracle("a=np.take(np.arange(10.),[-1,-2,0,3])");
  if (o) CHECK(allclose(take(a, idx), *o, 0, 0, true));
}

TEST_CASE("mined multiarray: put (flat, C order) vs numpy") {
  ndarray a = arange(0.0, 6.0, 1.0).reshape({2, 3});
  put(a, i64({0, 4, 5}), arange(100.0, 103.0, 1.0));
  auto o = npt::oracle("a=np.arange(6.).reshape(2,3); np.put(a,[0,4,5],[100,101,102.]); a=a");
  if (o) CHECK(allclose(a, *o, 0, 0, true));
}

TEST_CASE("mined multiarray: place (masked assign, cycling values) vs numpy") {
  ndarray a = arange(0.0, 6.0, 1.0);
  ndarray m = bvec({true, false, true, true, false, true});
  place(a, m, arange(10.0, 12.0, 1.0));  // values cycle: [10,11]
  auto o = npt::oracle(
      "a=np.arange(6.); np.place(a,np.array([True,False,True,True,False,True]),[10,11.]); a=a");
  if (o) CHECK(allclose(a, *o, 0, 0, true));
}

TEST_CASE("mined multiarray: choose vs numpy") {
  ndarray sel = i64({0, 1, 2, 1, 0});
  ndarray c0 = arange(0.0, 5.0, 1.0);
  ndarray c1 = arange(10.0, 15.0, 1.0);
  ndarray c2 = arange(20.0, 25.0, 1.0);
  auto o = npt::oracle(
      "a=np.choose([0,1,2,1,0],[np.arange(5.),np.arange(10,15.),np.arange(20,25.)])");
  if (o) CHECK(allclose(choose(sel, {c0, c1, c2}), *o, 0, 0, true));
}

TEST_CASE("mined multiarray: compress along an axis vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4});
  ndarray cond = bvec({true, false, true, false});
  auto o = npt::oracle(
      "A=np.arange(12.).reshape(3,4); a=np.compress([True,False,True,False],A,axis=1)");
  if (o) CHECK(allclose(compress(cond, A, int64_t{1}), *o, 0, 0, true));
}

TEST_CASE("mined multiarray: ravel_multi_index / unravel_index vs numpy") {
  auto orv = npt::oracle("a=np.ravel_multi_index(([1,2,0],[2,1,3]),(3,4))");
  if (orv) {
    ndarray r = ravel_multi_index({i64({1, 2, 0}), i64({2, 1, 3})}, Shape{3, 4});
    CHECK(allclose(r.astype(kFloat64), orv->astype(kFloat64), 0, 0, true));
  }
  auto ouv = npt::oracle("a=np.array(np.unravel_index([6,9,3],(3,4)))");
  if (ouv) {
    std::vector<ndarray> u = unravel_index(i64({6, 9, 3}), Shape{3, 4});
    ndarray got(Shape{2, 3}, kInt64, Order::C);
    for (int r = 0; r < 2; ++r) for (int c = 0; c < 3; ++c) got.set_item<int64_t>({r, c}, u[r].item<int64_t>({c}));
    CHECK(allclose(got.astype(kFloat64), ouv->astype(kFloat64), 0, 0, true));
  }
}

TEST_CASE("mined multiarray: 0-d array behaviors vs numpy") {
  ndarray z(Shape{}, kFloat64, Order::C);
  z.set_item<double>({}, 7.0);
  CHECK((z.shape() == Shape{}));
  CHECK(z.ndim() == 0);
  CHECK(z.size() == 1);
  CHECK(z.item<double>({}) == 7.0);
  auto o = npt::oracle("a=np.array(np.sum(np.array(7.0)))");  // sum of a 0-d array
  if (o) CHECK(allclose(sum(z), *o, 0, 0, true));
}
