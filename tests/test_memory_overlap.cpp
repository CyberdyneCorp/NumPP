#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cstdint>

using namespace numpp;

TEST_CASE("array shares memory with its own slice view vs numpy") {
  auto a = arange(0.0, 10.0, 1.0, kFloat64);
  auto v = a.index({Slice{int64_t{2}, int64_t{8}, 1}});
  // numpy: np.shares_memory(a, a[2:8]) is True
  CHECK(shares_memory(a, v));
  CHECK(may_share_memory(a, v));

  if (!npt::numpy_available()) return;
  auto o = npt::oracle(
      "x=np.arange(10.); a=np.array(np.shares_memory(x, x[2:8]))");
  if (!o) return;
  CHECK((shares_memory(a, v) == (o->item<bool>({}))));
}

TEST_CASE("array does not share memory with its copy vs numpy") {
  auto a = arange(0.0, 10.0, 1.0, kFloat64);
  auto c = a.copy();
  // numpy: np.shares_memory(a, a.copy()) is False
  CHECK(!shares_memory(a, c));
  CHECK(!may_share_memory(a, c));

  if (!npt::numpy_available()) return;
  auto o = npt::oracle(
      "x=np.arange(10.); a=np.array(np.shares_memory(x, x.copy()))");
  if (!o) return;
  CHECK((shares_memory(a, c) == (o->item<bool>({}))));
}

TEST_CASE("disjoint slices do not share memory vs numpy") {
  auto a = arange(0.0, 6.0, 1.0, kFloat64);
  auto lo = a.index({Slice{std::nullopt, int64_t{2}, 1}});   // a[:2]
  auto hi = a.index({Slice{int64_t{3}, std::nullopt, 1}});   // a[3:]
  // numpy: np.shares_memory(a[:2], a[3:]) is False (no byte overlap)
  CHECK(!shares_memory(lo, hi));
  CHECK(!may_share_memory(lo, hi));

  if (!npt::numpy_available()) return;
  auto o = npt::oracle(
      "x=np.arange(6.); a=np.array(np.shares_memory(x[:2], x[3:]))");
  if (!o) return;
  CHECK((shares_memory(lo, hi) == (o->item<bool>({}))));
}

TEST_CASE("overlapping slices share memory vs numpy") {
  auto a = arange(0.0, 6.0, 1.0, kFloat64);
  auto lo = a.index({Slice{std::nullopt, int64_t{4}, 1}});  // a[:4]
  auto hi = a.index({Slice{int64_t{2}, std::nullopt, 1}});  // a[2:]
  // numpy: np.may_share_memory(a[:4], a[2:]) is True (byte ranges overlap)
  CHECK(may_share_memory(lo, hi));

  if (!npt::numpy_available()) return;
  auto o = npt::oracle(
      "x=np.arange(6.); a=np.array(np.may_share_memory(x[:4], x[2:]))");
  if (!o) return;
  CHECK((may_share_memory(lo, hi) == (o->item<bool>({}))));
}

TEST_CASE("transpose view shares memory vs numpy") {
  auto a = arange(0.0, 12.0, 1.0, kFloat64).reshape({3, 4});
  auto t = a.transpose();
  // numpy: np.shares_memory(a, a.T) is True
  CHECK(shares_memory(a, t));
  CHECK(may_share_memory(a, t));

  if (!npt::numpy_available()) return;
  auto o = npt::oracle(
      "x=np.arange(12.).reshape(3,4); a=np.array(np.shares_memory(x, x.T))");
  if (!o) return;
  CHECK((shares_memory(a, t) == (o->item<bool>({}))));
}

TEST_CASE("two independent arrays do not share memory") {
  auto a = arange(0.0, 5.0, 1.0, kFloat64);
  auto b = arange(0.0, 5.0, 1.0, kFloat64);
  // Different buffers -> never share.
  CHECK(!shares_memory(a, b));
  CHECK(!may_share_memory(a, b));
}
