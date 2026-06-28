// Cases mined from numpy's numpy/_core/tests/test_shape_base.py and
// lib/tests/test_shape_base.py: array_split (even and uneven), tile, repeat,
// and the stacking helpers (stack/concatenate/vstack/hstack/dstack/column_stack).
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <string>
#include <vector>

using namespace numpp;

namespace {
ndarray ar(int64_t n) { return arange(0.0, static_cast<double>(n), 1.0); }
ndarray row(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined shape_base: array_split uneven division vs numpy") {
  // numpy splits 7 into [3,2,2]; each returned piece must match.
  std::vector<ndarray> parts = array_split(ar(7), 3);
  CHECK(parts.size() == 3);
  for (size_t i = 0; i < parts.size(); ++i) {
    auto o = npt::oracle("a=np.array_split(np.arange(7.),3)[" + std::to_string(i) + "]");
    if (o) CHECK(allclose(parts[i], *o, 1e-12, 1e-12, true));
  }
}

TEST_CASE("mined shape_base: array_split even division vs numpy") {
  std::vector<ndarray> parts = array_split(ar(6), 3);
  CHECK(parts.size() == 3);
  for (size_t i = 0; i < parts.size(); ++i) {
    auto o = npt::oracle("a=np.array_split(np.arange(6.),3)[" + std::to_string(i) + "]");
    if (o) CHECK(allclose(parts[i], *o, 1e-12, 1e-12, true));
  }
}

TEST_CASE("mined shape_base: tile (1-D and 2-D reps) vs numpy") {
  auto o1 = npt::oracle("a=np.tile([1,2,3.],3)");
  if (o1) CHECK(allclose(tile(row({1, 2, 3}), {3}), *o1, 1e-12, 1e-12, true));
  ndarray m = ar(4).reshape({2, 2});
  auto o2 = npt::oracle("a=np.tile(np.arange(4.).reshape(2,2),(2,2))");
  if (o2) CHECK(allclose(tile(m, {2, 2}), *o2, 1e-12, 1e-12, true));
}

TEST_CASE("mined shape_base: repeat (flat and per-axis) vs numpy") {
  auto o1 = npt::oracle("a=np.repeat([1,2,3.],2)");
  if (o1) CHECK(allclose(repeat(row({1, 2, 3}), 2), *o1, 1e-12, 1e-12, true));
  ndarray m = ar(4).reshape({2, 2});
  auto o2 = npt::oracle("a=np.repeat(np.arange(4.).reshape(2,2),2,axis=0)");
  if (o2) CHECK(allclose(repeat(m, 2, 0), *o2, 1e-12, 1e-12, true));
  auto o3 = npt::oracle("a=np.repeat(np.arange(4.).reshape(2,2),3,axis=1)");
  if (o3) CHECK(allclose(repeat(m, 3, 1), *o3, 1e-12, 1e-12, true));
}

TEST_CASE("mined shape_base: stack / concatenate axis variants vs numpy") {
  ndarray a = row({1, 2, 3}), b = row({4, 5, 6});
  auto o0 = npt::oracle("a=np.stack([[1,2,3.],[4,5,6.]],axis=0)");
  if (o0) CHECK(allclose(stack({a, b}, 0), *o0, 1e-12, 1e-12, true));
  auto o1 = npt::oracle("a=np.stack([[1,2,3.],[4,5,6.]],axis=1)");
  if (o1) CHECK(allclose(stack({a, b}, 1), *o1, 1e-12, 1e-12, true));
  auto oc = npt::oracle("a=np.concatenate([[1,2,3.],[4,5,6.]])");
  if (oc) CHECK(allclose(concatenate({a, b}, 0), *oc, 1e-12, 1e-12, true));
}

TEST_CASE("mined shape_base: vstack / hstack / dstack / column_stack vs numpy") {
  ndarray a = row({1, 2, 3}), b = row({4, 5, 6});
  auto ov = npt::oracle("a=np.vstack(([1,2,3.],[4,5,6.]))");
  if (ov) CHECK(allclose(vstack({a, b}), *ov, 1e-12, 1e-12, true));
  auto oh = npt::oracle("a=np.hstack(([1,2,3.],[4,5,6.]))");
  if (oh) CHECK(allclose(hstack({a, b}), *oh, 1e-12, 1e-12, true));
  auto od = npt::oracle("a=np.dstack(([1,2,3.],[4,5,6.]))");
  if (od) CHECK(allclose(dstack({a, b}), *od, 1e-12, 1e-12, true));
  auto oc = npt::oracle("a=np.column_stack(([1,2,3.],[4,5,6.]))");
  if (oc) CHECK(allclose(column_stack({a, b}), *oc, 1e-12, 1e-12, true));
}
