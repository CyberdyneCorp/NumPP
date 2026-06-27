#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <vector>

using namespace numpp;

namespace {
ndarray mk(double n, std::vector<int64_t> shape) {
  return arange(0.0, n, 1.0).reshape(Shape(shape.begin(), shape.end()));
}
}  // namespace

TEST_CASE("einsum ellipsis batched matmul vs numpy") {
  ndarray a = mk(24, {2, 3, 4}), b = mk(40, {2, 4, 5});
  ndarray got = einsum("...ij,...jk->...ik", {a, b});
  CHECK((got.shape() == Shape{2, 3, 5}));
  auto o = npt::oracle(
      "A=np.arange(24.).reshape(2,3,4); B=np.arange(40.).reshape(2,4,5); "
      "a=np.einsum('...ij,...jk->...ik',A,B)");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}

TEST_CASE("einsum ellipsis broadcasts a lower-rank operand vs numpy") {
  ndarray a = mk(24, {2, 3, 4}), b = mk(20, {4, 5});  // b has no batch axis
  ndarray got = einsum("...ij,...jk->...ik", {a, b});
  CHECK((got.shape() == Shape{2, 3, 5}));
  auto o = npt::oracle(
      "A=np.arange(24.).reshape(2,3,4); B=np.arange(20.).reshape(4,5); "
      "a=np.einsum('...ij,...jk->...ik',A,B)");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}

TEST_CASE("einsum ellipsis batched diagonal vs numpy") {
  ndarray a = mk(48, {3, 4, 4});
  ndarray got = einsum("...ii->...i", {a});
  CHECK((got.shape() == Shape{3, 4}));
  auto o = npt::oracle("A=np.arange(48.).reshape(3,4,4); a=np.einsum('...ii->...i',A)");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}

TEST_CASE("einsum ellipsis implicit output vs numpy") {
  ndarray a = mk(24, {2, 3, 4}), b = mk(40, {2, 4, 5});
  ndarray got = einsum("...ij,...jk", {a, b});  // implicit -> ...ik
  auto o = npt::oracle(
      "A=np.arange(24.).reshape(2,3,4); B=np.arange(40.).reshape(2,4,5); "
      "a=np.einsum('...ij,...jk',A,B)");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}

TEST_CASE("einsum ellipsis transpose of last two axes vs numpy") {
  ndarray a = mk(48, {2, 2, 3, 4});
  ndarray got = einsum("...ij->...ji", {a});
  CHECK((got.shape() == Shape{2, 2, 4, 3}));
  auto o = npt::oracle("A=np.arange(48.).reshape(2,2,3,4); a=np.einsum('...ij->...ji',A)");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}

TEST_CASE("einsum ellipsis with zero batch axes vs numpy") {
  ndarray a = mk(12, {3, 4}), b = mk(20, {4, 5});
  ndarray got = einsum("...ij,...jk->...ik", {a, b});
  CHECK((got.shape() == Shape{3, 5}));
  auto o = npt::oracle(
      "A=np.arange(12.).reshape(3,4); B=np.arange(20.).reshape(4,5); "
      "a=np.einsum('...ij,...jk->...ik',A,B)");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}

TEST_CASE("einsum ellipsis with optimize matches non-optimized") {
  ndarray a = mk(24, {2, 3, 4}), b = mk(40, {2, 4, 5}), c = mk(30, {2, 5, 3});
  ndarray plain = einsum("...ij,...jk,...kl->...il", {a, b, c});
  ndarray opt = einsum("...ij,...jk,...kl->...il", {a, b, c}, true);
  CHECK(allclose(opt, plain, 1e-8, 1e-10, true));
}
