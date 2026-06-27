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

TEST_CASE("einsum optimize matches the unoptimized result (two operands)") {
  ndarray A = mk(6, {2, 3}), B = mk(12, {3, 4});
  ndarray plain = einsum("ij,jk->ik", {A, B});
  ndarray opt = einsum("ij,jk->ik", {A, B}, true);
  CHECK(allclose(opt, plain, 1e-10, 1e-12, true));
}

TEST_CASE("einsum optimize chain ij,jk,kl->il vs numpy") {
  ndarray A = mk(6, {2, 3}), B = mk(12, {3, 4}), C = mk(8, {4, 2});
  ndarray plain = einsum("ij,jk,kl->il", {A, B, C});
  ndarray opt = einsum("ij,jk,kl->il", {A, B, C}, true);
  CHECK(allclose(opt, plain, 1e-9, 1e-11, true));
  auto o = npt::oracle(
      "A=np.arange(6.).reshape(2,3); B=np.arange(12.).reshape(3,4); "
      "C=np.arange(8.).reshape(4,2); a=np.einsum('ij,jk,kl->il',A,B,C,optimize=True)");
  if (o) CHECK(allclose(opt, *o, 1e-9, 1e-11, true));
}

TEST_CASE("einsum optimize 4-operand chain vs numpy") {
  ndarray A = mk(6, {2, 3}), B = mk(12, {3, 4}), C = mk(8, {4, 2}), D = mk(10, {2, 5});
  ndarray opt = einsum("ij,jk,kl,lm->im", {A, B, C, D}, true);
  auto o = npt::oracle(
      "A=np.arange(6.).reshape(2,3); B=np.arange(12.).reshape(3,4); "
      "C=np.arange(8.).reshape(4,2); D=np.arange(10.).reshape(2,5); "
      "a=np.einsum('ij,jk,kl,lm->im',A,B,C,D,optimize=True)");
  if (o) CHECK(allclose(opt, *o, 1e-8, 1e-10, true));
}

TEST_CASE("einsum_path returns one step per contraction") {
  ndarray A = mk(6, {2, 3}), B = mk(12, {3, 4}), C = mk(8, {4, 2});
  auto path = einsum_path("ij,jk,kl->il", {A, B, C});
  CHECK(static_cast<int64_t>(path.size()) == 2);  // nops - 1
  for (const auto& step : path) CHECK(step.size() == 2);
}

TEST_CASE("einsum optimize inner product to scalar vs numpy") {
  ndarray a = arange(0.0, 4.0, 1.0), b = arange(1.0, 5.0, 1.0);
  ndarray opt = einsum("i,i->", {a, b}, true);
  auto o = npt::oracle("a=np.einsum('i,i->',np.arange(4.),np.arange(1,5.),optimize=True)");
  if (o) CHECK(allclose(opt, *o, 1e-10, 1e-12, true));
}
