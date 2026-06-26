#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <vector>

using namespace numpp;

// ---- einsum ----
TEST_CASE("einsum matmul vs numpy") {
  ndarray a = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray b = arange(0., 12., 1., kFloat64).reshape({3, 4});
  auto o = npt::oracle("a=np.einsum('ij,jk->ik',np.arange(6.).reshape(2,3),np.arange(12.).reshape(3,4))");
  if (o) CHECK(allclose(einsum("ij,jk->ik", {a, b}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("einsum trace vs numpy") {
  ndarray a = arange(0., 9., 1., kFloat64).reshape({3, 3});
  auto o = npt::oracle("a=np.einsum('ii->',np.arange(9.).reshape(3,3))");
  if (o) CHECK(allclose(einsum("ii->", {a}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("einsum diagonal vs numpy") {
  ndarray a = arange(0., 9., 1., kFloat64).reshape({3, 3});
  auto o = npt::oracle("a=np.einsum('ii->i',np.arange(9.).reshape(3,3))");
  if (o) CHECK(allclose(einsum("ii->i", {a}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("einsum transpose vs numpy") {
  ndarray a = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.einsum('ij->ji',np.arange(6.).reshape(2,3))");
  if (o) CHECK(allclose(einsum("ij->ji", {a}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("einsum implicit sum vs numpy") {
  ndarray a = arange(0., 6., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.einsum('ij->',np.arange(6.).reshape(2,3))");
  if (o) CHECK(allclose(einsum("ij->", {a}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("einsum outer (implicit out) vs numpy") {
  ndarray a = arange(1., 4., 1., kFloat64);
  ndarray b = arange(1., 5., 1., kFloat64);
  auto o = npt::oracle("a=np.einsum('i,j',np.arange(1,4.),np.arange(1,5.))");
  if (o) CHECK(allclose(einsum("i,j", {a, b}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("einsum batched matmul vs numpy") {
  ndarray a = arange(0., 24., 1., kFloat64).reshape({2, 3, 4});
  ndarray b = arange(0., 40., 1., kFloat64).reshape({2, 4, 5});
  auto o = npt::oracle("a=np.einsum('bij,bjk->bik',np.arange(24.).reshape(2,3,4),np.arange(40.).reshape(2,4,5))");
  if (o) CHECK(allclose(einsum("bij,bjk->bik", {a, b}), *o, 1e-9, 1e-12, true));
}
TEST_CASE("einsum inner product vs numpy") {
  ndarray a = arange(1., 5., 1., kFloat64);
  ndarray b = arange(2., 6., 1., kFloat64);
  auto o = npt::oracle("a=np.einsum('i,i->',np.arange(1,5.),np.arange(2,6.))");
  if (o) CHECK(allclose(einsum("i,i->", {a, b}), *o, 1e-9, 1e-12, true));
}

// ---- tensordot ----
TEST_CASE("tensordot n2 vs numpy") {
  ndarray a = arange(0., 24., 1., kFloat64).reshape({2, 3, 4});
  ndarray b = arange(0., 24., 1., kFloat64).reshape({3, 4, 2});
  auto o = npt::oracle("a=np.tensordot(np.arange(24.).reshape(2,3,4),np.arange(24.).reshape(3,4,2))");
  if (o) CHECK(allclose(tensordot(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("tensordot axes vs numpy") {
  ndarray a = arange(0., 12., 1., kFloat64).reshape({3, 4});
  ndarray b = arange(0., 12., 1., kFloat64).reshape({4, 3});
  auto o = npt::oracle("a=np.tensordot(np.arange(12.).reshape(3,4),np.arange(12.).reshape(4,3),axes=([1],[0]))");
  if (o) CHECK(allclose(tensordot(a, b, {1}, {0}), *o, 1e-9, 1e-12, true));
}

// ---- cross ----
TEST_CASE("cross 3d vs numpy") {
  ndarray a = arange(1., 4., 1., kFloat64);
  ndarray b = arange(4., 7., 1., kFloat64);
  auto o = npt::oracle("a=np.cross([1,2,3],[4,5,6])");
  if (o) CHECK(allclose(cross(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("cross 2d scalar vs numpy") {
  ndarray a = arange(1., 3., 1., kFloat64);
  ndarray b = arange(3., 5., 1., kFloat64);
  auto o = npt::oracle("a=np.cross([1,2],[3,4])");
  if (o) CHECK(allclose(cross(a, b), *o, 1e-9, 1e-12, true));
}
TEST_CASE("cross batched vs numpy") {
  ndarray a = arange(1., 7., 1., kFloat64).reshape({2, 3});
  ndarray b = arange(7., 13., 1., kFloat64).reshape({2, 3});
  auto o = npt::oracle("a=np.cross(np.arange(1,7.).reshape(2,3),np.arange(7,13.).reshape(2,3))");
  if (o) CHECK(allclose(cross(a, b), *o, 1e-9, 1e-12, true));
}

// ---- cond / multi_dot ----
TEST_CASE("cond vs numpy") {
  ndarray a = arange(1., 5., 1., kFloat64).reshape({2, 2});
  a.set_item<double>({1, 1}, 5.0);
  auto o = npt::oracle("a=np.linalg.cond(np.array([[1.,2.],[3.,5.]]))");
  if (o) CHECK(allclose(cond(a), *o, 1e-7, 1e-9, true));
}
TEST_CASE("multi_dot vs numpy") {
  ndarray a = arange(0., 6., 1., kFloat64).reshape({2, 3});
  ndarray b = arange(0., 12., 1., kFloat64).reshape({3, 4});
  ndarray c = arange(0., 4., 1., kFloat64).reshape({4, 1});
  auto o = npt::oracle(
      "a=np.linalg.multi_dot([np.arange(6.).reshape(2,3),np.arange(12.).reshape(3,4),np.arange(4.).reshape(4,1)])");
  if (o) CHECK(allclose(multi_dot({a, b, c}), *o, 1e-9, 1e-12, true));
}
