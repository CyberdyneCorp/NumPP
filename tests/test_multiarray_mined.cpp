// Cases mined from numpy's numpy/_core/tests/test_multiarray.py: array-core
// behaviors — argmax/argmin (incl. NaN), negative-step slicing, clip, diagonal
// offsets, repeat, take, concatenate, broadcasting, transpose — vs the oracle.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <limits>
#include <optional>
#include <vector>

using namespace numpp;

namespace {
const double NAN_ = std::numeric_limits<double>::quiet_NaN();
ndarray dval(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray i64(const std::vector<int64_t>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int64_t>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined multiarray: argmax/argmin pick the NaN index vs numpy (TestArgmax)") {
  ndarray x = dval({1, NAN_, 2});
  auto omax = npt::oracle("a=np.array(np.argmax([1,np.nan,2]))");
  auto omin = npt::oracle("a=np.array(np.argmin([1,np.nan,2]))");
  if (omax) CHECK(argmax(x).item<int64_t>({}) == omax->astype(kInt64).item<int64_t>({}));
  if (omin) CHECK(argmin(x).item<int64_t>({}) == omin->astype(kInt64).item<int64_t>({}));
}

TEST_CASE("mined multiarray: argmax over an axis vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4});
  auto o = npt::oracle("A=np.arange(12.).reshape(3,4); a=np.argmax(A,axis=1)");
  if (o) CHECK(allclose(argmax(A, int64_t{1}).astype(kFloat64), o->astype(kFloat64), 0, 0, true));
}

TEST_CASE("mined multiarray: negative-step slicing vs numpy") {
  ndarray a = arange(0.0, 10.0, 1.0);
  auto rev = a.index({Slice{std::nullopt, std::nullopt, -1}});      // a[::-1]
  auto mid = a.index({Slice{int64_t{8}, int64_t{2}, -2}});          // a[8:2:-2]
  auto o1 = npt::oracle("a=np.arange(10.)[::-1]");
  auto o2 = npt::oracle("a=np.arange(10.)[8:2:-2]");
  if (o1) CHECK(allclose(rev, *o1, 0, 0, true));
  if (o2) CHECK(allclose(mid, *o2, 0, 0, true));
}

TEST_CASE("mined multiarray: 2-D reversed slicing vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4});
  auto rr = A.index({Slice{std::nullopt, std::nullopt, -1}, Slice{std::nullopt, std::nullopt, -1}});
  auto o = npt::oracle("A=np.arange(12.).reshape(3,4); a=A[::-1,::-1]");
  if (o) CHECK(allclose(rr, *o, 0, 0, true));
}

TEST_CASE("mined multiarray: clip (incl. NaN bounds passthrough) vs numpy") {
  ndarray a = dval({-3, -1, 0, 2, 5, NAN_});
  ndarray lo = dval({-1, -1, -1, -1, -1, -1});
  ndarray hi = dval({3, 3, 3, 3, 3, 3});
  auto o = npt::oracle("a=np.clip(np.array([-3,-1,0,2,5,np.nan]),-1,3)");
  if (o) CHECK(allclose(clip(a, lo, hi), *o, 1e-12, 1e-12, true));
}

TEST_CASE("mined multiarray: diagonal with offsets vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4});
  const char* PYA = "A=np.arange(12.).reshape(3,4); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYA) + "a=" + e);
    if (o) CHECK(allclose(got, *o, 0, 0, true));
  };
  chk(diagonal(A, 0), "np.diagonal(A,0)");
  chk(diagonal(A, 1), "np.diagonal(A,1)");
  chk(diagonal(A, -1), "np.diagonal(A,-1)");
}

TEST_CASE("mined multiarray: repeat (scalar + per-element) vs numpy") {
  ndarray a = i64({1, 2, 3});
  auto o1 = npt::oracle("a=np.repeat([1,2,3],2)");
  if (o1) CHECK(allclose(repeat(a, 2).astype(kFloat64), o1->astype(kFloat64), 0, 0, true));
  ndarray A = arange(0.0, 6.0, 1.0).reshape({2, 3});
  auto o2 = npt::oracle("A=np.arange(6.).reshape(2,3); a=np.repeat(A,2,axis=1)");
  if (o2) CHECK(allclose(repeat(A, 2, int64_t{1}), *o2, 0, 0, true));
}

TEST_CASE("mined multiarray: take along an axis vs numpy") {
  ndarray A = arange(0.0, 12.0, 1.0).reshape({3, 4});
  ndarray idx = i64({0, 2, 3});
  auto o = npt::oracle("A=np.arange(12.).reshape(3,4); a=np.take(A,[0,2,3],axis=1)");
  if (o) CHECK(allclose(take(A, idx, int64_t{1}), *o, 0, 0, true));
}

TEST_CASE("mined multiarray: concatenate / stack along axes vs numpy") {
  ndarray A = arange(0.0, 6.0, 1.0).reshape({2, 3});
  ndarray B = arange(6.0, 12.0, 1.0).reshape({2, 3});
  auto oc = npt::oracle(
      "A=np.arange(6.).reshape(2,3); B=np.arange(6,12.).reshape(2,3); a=np.concatenate([A,B],axis=1)");
  auto os = npt::oracle(
      "A=np.arange(6.).reshape(2,3); B=np.arange(6,12.).reshape(2,3); a=np.stack([A,B],axis=0)");
  if (oc) CHECK(allclose(concatenate({A, B}, 1), *oc, 0, 0, true));
  if (os) CHECK(allclose(stack({A, B}, 0), *os, 0, 0, true));
}

TEST_CASE("mined multiarray: broadcast_to / transpose(axes) vs numpy") {
  ndarray a = i64({1, 2, 3}).reshape({1, 3});
  auto ob = npt::oracle("a=np.broadcast_to(np.array([[1,2,3]]),(4,3))");
  if (ob) CHECK(allclose(a.broadcast_to(Shape{4, 3}).astype(kFloat64), ob->astype(kFloat64), 0, 0, true));
  ndarray A = arange(0.0, 24.0, 1.0).reshape({2, 3, 4});
  auto ot = npt::oracle("A=np.arange(24.).reshape(2,3,4); a=np.transpose(A,(1,2,0))");
  if (ot) CHECK(allclose(A.transpose({1, 2, 0}), *ot, 0, 0, true));
}
