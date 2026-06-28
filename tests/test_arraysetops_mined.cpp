// Cases mined from numpy's numpy/lib/tests/test_arraysetops.py: unique (values,
// counts, index, inverse), intersect1d, union1d, setdiff1d, isin — vs the oracle.
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

TEST_CASE("mined arraysetops: unique values and counts vs numpy") {
  ndarray a = i64v({3, 1, 2, 1, 3, 3});
  auto ov = npt::oracle("a=np.unique([3,1,2,1,3,3])");
  if (ov) CHECK(allclose(unique(a).astype(kFloat64), *ov, 0, 0, true));
  auto oc = npt::oracle("a=np.unique([3,1,2,1,3,3],return_counts=True)[1]");
  if (oc) CHECK(allclose(unique_ex(a, false, false, true).counts.astype(kFloat64), *oc, 0, 0, true));
}

TEST_CASE("mined arraysetops: unique return_index / return_inverse vs numpy") {
  ndarray a = i64v({3, 1, 2, 1, 3, 3});
  auto oi = npt::oracle("a=np.unique([3,1,2,1,3,3],return_index=True)[1]");
  if (oi) CHECK(allclose(unique_ex(a, true, false, false).index.astype(kFloat64), *oi, 0, 0, true));
  auto ov = npt::oracle("a=np.unique([3,1,2,1,3,3],return_inverse=True)[1].ravel()");
  if (ov) CHECK(allclose(unique_ex(a, false, true, false).inverse.astype(kFloat64), *ov, 0, 0, true));
}

TEST_CASE("mined arraysetops: intersect1d / union1d / setdiff1d vs numpy") {
  ndarray a = i64v({1, 3, 4, 3}), b = i64v({3, 1, 2, 1});
  auto oi = npt::oracle("a=np.intersect1d([1,3,4,3],[3,1,2,1])");
  if (oi) CHECK(allclose(intersect1d(a, b).astype(kFloat64), *oi, 0, 0, true));

  ndarray c = i64v({-1, 0, 1}), d = i64v({1, 2, 3});
  auto ou = npt::oracle("a=np.union1d([-1,0,1],[1,2,3])");
  if (ou) CHECK(allclose(union1d(c, d).astype(kFloat64), *ou, 0, 0, true));

  ndarray e = i64v({1, 2, 3, 4}), f = i64v({2, 4});
  auto os = npt::oracle("a=np.setdiff1d([1,2,3,4],[2,4])");
  if (os) CHECK(allclose(setdiff1d(e, f).astype(kFloat64), *os, 0, 0, true));
}

TEST_CASE("mined arraysetops: isin membership mask vs numpy") {
  ndarray a = i64v({1, 2, 5, 0});
  ndarray b = i64v({0, 2, 4});
  auto o = npt::oracle("a=np.isin([1,2,5,0],[0,2,4])");
  if (o) CHECK(allclose(isin(a, b).astype(kFloat64), *o, 0, 0, true));
}
