#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cstdint>
#include <utility>
#include <vector>

using namespace numpp;

TEST_CASE("ndindex empty shape yields one empty index") {
  auto idxs = ndindex(Shape{});
  CHECK(idxs.size() == 1);
  CHECK(idxs[0].empty());
}

TEST_CASE("ndindex zero-length axis yields no indices") {
  auto idxs = ndindex(Shape{2, 0});
  CHECK(idxs.empty());
}

TEST_CASE("ndindex C order count and ordering") {
  auto idxs = ndindex(Shape{2, 3});
  CHECK(idxs.size() == 6);
  CHECK((idxs.front() == std::vector<int64_t>{0, 0}));
  CHECK((idxs[1] == std::vector<int64_t>{0, 1}));
  CHECK((idxs.back() == std::vector<int64_t>{1, 2}));
}

TEST_CASE("ndindex 2x3 flattened vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle("a = np.array(list(np.ndindex(2,3)))");
  if (!o) return;
  auto idxs = ndindex(Shape{2, 3});
  ndarray got(Shape{static_cast<int64_t>(idxs.size()), 2}, kFloat64);
  for (int64_t i = 0; i < static_cast<int64_t>(idxs.size()); ++i)
    for (int64_t j = 0; j < 2; ++j)
      got.set_item<double>({i, j},
          static_cast<double>(idxs[static_cast<size_t>(i)][static_cast<size_t>(j)]));
  CHECK(allclose(got, *o, 1e-9, 1e-12, true));
}

TEST_CASE("nditer transposed non-contiguous view C order vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(
      "x = np.arange(6).reshape(2,3).T; a = np.array(list(np.nditer(x, order='C')))");
  if (!o) return;
  auto x = arange(0.0, 6.0, 1.0, kFloat64).reshape({2, 3}).transpose();
  CHECK((x.shape() == Shape{3, 2}));
  CHECK(!(x.c_contiguous()));
  CHECK(allclose(nditer(x), *o, 1e-9, 1e-12, true));
}

TEST_CASE("nditer broadcasted view C order vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle(
      "x = np.broadcast_to(np.arange(3.), (2,3)); a = np.array(list(np.nditer(x, order='C')))");
  if (!o) return;
  auto x = arange(0.0, 3.0, 1.0, kFloat64).broadcast_to({2, 3});
  CHECK(allclose(nditer(x), *o, 1e-9, 1e-12, true));
}

TEST_CASE("ndenumerate index and value order") {
  auto x = arange(0.0, 6.0, 1.0, kFloat64).reshape({2, 3});
  auto p = ndenumerate(x);
  CHECK(p.size() == 6);
  CHECK((p[0].first == std::vector<int64_t>{0, 0}));
  CHECK(p[0].second == 0.0);
  CHECK((p[5].first == std::vector<int64_t>{1, 2}));
  CHECK(p[5].second == 5.0);
}

TEST_CASE("ndenumerate values C order vs numpy") {
  if (!npt::numpy_available()) return;
  auto o = npt::oracle("x = np.arange(6).reshape(2,3).T; a = x.reshape(-1)");
  if (!o) return;
  auto x = arange(0.0, 6.0, 1.0, kFloat64).reshape({2, 3}).transpose();
  auto pairs = ndenumerate(x);
  ndarray got(Shape{static_cast<int64_t>(pairs.size())}, kFloat64);
  for (int64_t i = 0; i < static_cast<int64_t>(pairs.size()); ++i)
    got.set_item<double>({i}, pairs[static_cast<size_t>(i)].second);
  CHECK(allclose(got, *o, 1e-9, 1e-12, true));
}

TEST_CASE("ndenumerate rejects complex dtype") {
  ndarray c(Shape{2}, kComplex128);
  CHECK_THROWS_AS(ndenumerate(c), type_error);
}

TEST_CASE("nditer rejects complex dtype") {
  ndarray c(Shape{2}, kComplex128);
  CHECK_THROWS_AS(nditer(c), type_error);
}
