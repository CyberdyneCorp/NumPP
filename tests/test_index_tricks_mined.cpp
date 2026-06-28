// Cases mined from numpy's numpy/lib/tests/test_index_tricks.py: ravel_multi_index
// and unravel_index (C order), plus tril/triu/diag index generators with offsets.
// (numpp.ravel_multi_index/unravel_index are C-order only; order='F' is out of scope.)
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <string>
#include <vector>

using namespace numpp;

namespace {
ndarray i64v(const std::vector<int64_t>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int64_t>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("mined index_tricks: ravel_multi_index (C order) vs numpy") {
  ndarray r = i64v({1, 0, 1}), c = i64v({2, 1, 0});
  auto o = npt::oracle("a=np.ravel_multi_index(([1,0,1],[2,1,0]),(2,3))");
  if (o) CHECK(allclose(ravel_multi_index({r, c}, Shape{2, 3}).astype(kFloat64), *o, 0, 0, true));
}

TEST_CASE("mined index_tricks: unravel_index (C order) vs numpy") {
  ndarray idx = i64v({0, 3, 5});
  std::vector<ndarray> got = unravel_index(idx, Shape{2, 3});
  CHECK(got.size() == 2);
  auto orow = npt::oracle("a=np.unravel_index([0,3,5],(2,3))[0]");
  if (orow) CHECK(allclose(got[0].astype(kFloat64), *orow, 0, 0, true));
  auto ocol = npt::oracle("a=np.unravel_index([0,3,5],(2,3))[1]");
  if (ocol) CHECK(allclose(got[1].astype(kFloat64), *ocol, 0, 0, true));
}

TEST_CASE("mined index_tricks: tril_indices / triu_indices with offsets vs numpy") {
  std::vector<ndarray> tl = tril_indices(3);
  auto tlr = npt::oracle("a=np.tril_indices(3)[0]");
  if (tlr) CHECK(allclose(tl[0].astype(kFloat64), *tlr, 0, 0, true));
  auto tlc = npt::oracle("a=np.tril_indices(3)[1]");
  if (tlc) CHECK(allclose(tl[1].astype(kFloat64), *tlc, 0, 0, true));

  std::vector<ndarray> tu = triu_indices(3, 1);
  auto tur = npt::oracle("a=np.triu_indices(3,1)[0]");
  if (tur) CHECK(allclose(tu[0].astype(kFloat64), *tur, 0, 0, true));
  auto tuc = npt::oracle("a=np.triu_indices(3,1)[1]");
  if (tuc) CHECK(allclose(tu[1].astype(kFloat64), *tuc, 0, 0, true));
}

TEST_CASE("mined index_tricks: diag_indices vs numpy") {
  std::vector<ndarray> d = diag_indices(4);
  auto o0 = npt::oracle("a=np.diag_indices(4)[0]");
  if (o0) CHECK(allclose(d[0].astype(kFloat64), *o0, 0, 0, true));
  auto o1 = npt::oracle("a=np.diag_indices(4)[1]");
  if (o1) CHECK(allclose(d[1].astype(kFloat64), *o1, 0, 0, true));
}
