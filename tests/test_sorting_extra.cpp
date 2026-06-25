#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <complex>
#include <vector>

using namespace numpp;

namespace {
ndarray dvals(const std::vector<double>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray ivals(const std::vector<int64_t>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kInt64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<int64_t>()[i] = v[i];
  return a;
}
ndarray cvals(const std::vector<std::complex<double>>& v) {
  ndarray a({static_cast<int64_t>(v.size())}, kComplex128, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<std::complex<double>>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("lexsort vs numpy") {
  ndarray first = dvals({1, 5, 1, 5, 1});   // primary (last key)
  ndarray second = dvals({9, 4, 0, 4, 0});  // secondary (first key)
  auto o = npt::oracle("a=np.lexsort((np.array([9,4,0,4,0.]),np.array([1,5,1,5,1.])))");
  if (o) CHECK(allclose(lexsort({second, first}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("lexsort three keys vs numpy") {
  ndarray k0 = dvals({2, 1, 2, 1});
  ndarray k1 = dvals({0, 0, 1, 1});
  ndarray k2 = dvals({1, 1, 0, 0});  // primary
  auto o = npt::oracle(
      "a=np.lexsort((np.array([2,1,2,1.]),np.array([0,0,1,1.]),np.array([1,1,0,0.])))");
  if (o) CHECK(allclose(lexsort({k0, k1, k2}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("sort_complex vs numpy") {
  ndarray a = cvals({{3, -1}, {1, 2}, {1, -3}, {2, 0}, {1, 2}});
  auto o = npt::oracle("a=np.sort_complex([3-1j,1+2j,1-3j,2+0j,1+2j])");
  if (o) CHECK(allclose(sort_complex(a), *o, 1e-9, 1e-12, true));
}
TEST_CASE("sort_complex real input vs numpy") {
  ndarray a = dvals({3, 1, 4, 1, 5, 9, 2, 6});
  auto o = npt::oracle("a=np.sort_complex([3,1,4,1,5,9,2,6.])");
  if (o) CHECK(allclose(sort_complex(a), *o, 1e-9, 1e-12, true));
}
TEST_CASE("searchsorted with sorter vs numpy") {
  ndarray a = dvals({5, 1, 4, 2, 3});           // unsorted
  ndarray sorter = ivals({1, 3, 4, 2, 0});      // argsort(a)
  ndarray v = dvals({0, 2.5, 4, 6});
  auto o = npt::oracle(
      "a=np.searchsorted(np.array([5,1,4,2,3.]),[0,2.5,4,6],sorter=[1,3,4,2,0])");
  if (o) CHECK(allclose(searchsorted(a, v, "left", sorter), *o, 1e-9, 1e-12, true));
}
