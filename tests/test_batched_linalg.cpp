#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <algorithm>
#include <vector>

using namespace numpp;

namespace {
ndarray mkstack(const std::vector<double>& v, Shape shape) {
  ndarray a(shape, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
// A (3,2,2) stack of invertible matrices (each det = -2).
ndarray stack3() { return arange(0.0, 12.0, 1.0).reshape({3, 2, 2}); }
// Two SPD 2x2 matrices, shape (2,2,2).
ndarray spd2() { return mkstack({2, 0, 0, 3, 4, 1, 1, 3}, {2, 2, 2}); }
}  // namespace

TEST_CASE("batched det matches numpy") {
  ndarray A = stack3();
  CHECK((linalg::det(A).shape() == Shape{3}));
  auto o = npt::oracle("A=np.arange(12.).reshape(3,2,2); a=np.linalg.det(A)");
  if (o) CHECK(allclose(linalg::det(A), *o, 1e-9, 1e-11, true));
}

TEST_CASE("batched inv matches numpy") {
  ndarray A = stack3();
  CHECK((linalg::inv(A).shape() == Shape{3, 2, 2}));
  auto o = npt::oracle("A=np.arange(12.).reshape(3,2,2); a=np.linalg.inv(A)");
  if (o) CHECK(allclose(linalg::inv(A), *o, 1e-9, 1e-11, true));
}

TEST_CASE("batched solve with vector rhs matches numpy") {
  ndarray A = stack3();
  ndarray b = arange(1.0, 7.0, 1.0).reshape({3, 2});
  CHECK((linalg::solve(A, b).shape() == Shape{3, 2}));
  auto o = npt::oracle(
      "A=np.arange(12.).reshape(3,2,2); b=np.arange(1,7.).reshape(3,2); a=np.linalg.solve(A,b)");
  if (o) CHECK(allclose(linalg::solve(A, b), *o, 1e-9, 1e-11, true));
}

TEST_CASE("batched solve with matrix rhs matches numpy") {
  ndarray A = stack3();
  ndarray b = arange(1.0, 13.0, 1.0).reshape({3, 2, 2});
  CHECK((linalg::solve(A, b).shape() == Shape{3, 2, 2}));
  auto o = npt::oracle(
      "A=np.arange(12.).reshape(3,2,2); b=np.arange(1,13.).reshape(3,2,2); a=np.linalg.solve(A,b)");
  if (o) CHECK(allclose(linalg::solve(A, b), *o, 1e-9, 1e-11, true));
}

TEST_CASE("batched cholesky matches numpy") {
  ndarray S = spd2();
  auto o = npt::oracle(
      "S=np.array([[[2.,0],[0,3]],[[4,1],[1,3]]]); a=np.linalg.cholesky(S)");
  if (o) CHECK(allclose(linalg::cholesky(S), *o, 1e-9, 1e-11, true));
}

TEST_CASE("batched matrix_power matches numpy") {
  ndarray A = stack3();
  auto o = npt::oracle("A=np.arange(12.).reshape(3,2,2); a=np.linalg.matrix_power(A,3)");
  if (o) CHECK(allclose(linalg::matrix_power(A, 3), *o, 1e-7, 1e-9, true));
}

TEST_CASE("batched slogdet matches numpy") {
  ndarray A = stack3();
  linalg::SignLogDet sd = linalg::slogdet(A);
  CHECK((sd.sign.shape() == Shape{3}));
  auto os = npt::oracle("A=np.arange(12.).reshape(3,2,2); a=np.linalg.slogdet(A)[0]");
  auto ol = npt::oracle("A=np.arange(12.).reshape(3,2,2); a=np.linalg.slogdet(A)[1]");
  if (os) CHECK(allclose(sd.sign, *os, 1e-9, 1e-11, true));
  if (ol) CHECK(allclose(sd.logabsdet, *ol, 1e-9, 1e-11, true));
}

TEST_CASE("batched eigvalsh matches numpy (ascending)") {
  ndarray S = spd2();
  CHECK((linalg::eigvalsh(S).shape() == Shape{2, 2}));
  auto o = npt::oracle(
      "S=np.array([[[2.,0],[0,3]],[[4,1],[1,3]]]); a=np.linalg.eigvalsh(S)");
  if (o) CHECK(allclose(linalg::eigvalsh(S), *o, 1e-9, 1e-11, true));
}

TEST_CASE("batched svdvals matches numpy (descending)") {
  ndarray A = stack3();
  CHECK((linalg::svdvals(A).shape() == Shape{3, 2}));
  auto o = npt::oracle("A=np.arange(12.).reshape(3,2,2); a=np.linalg.svd(A,compute_uv=False)");
  if (o) CHECK(allclose(linalg::svdvals(A), *o, 1e-8, 1e-10, true));
}

TEST_CASE("batched pinv matches numpy") {
  ndarray A = stack3();
  auto o = npt::oracle("A=np.arange(12.).reshape(3,2,2); a=np.linalg.pinv(A)");
  if (o) CHECK(allclose(linalg::pinv(A), *o, 1e-7, 1e-9, true));
}

TEST_CASE("batched matrix_rank matches numpy") {
  ndarray A = stack3();
  CHECK((linalg::matrix_rank(A).shape() == Shape{3}));
  auto o = npt::oracle("A=np.arange(12.).reshape(3,2,2); a=np.linalg.matrix_rank(A)");
  if (o) CHECK(allclose(linalg::matrix_rank(A), *o, 0, 0, true));
}

TEST_CASE("batched qr reconstructs each matrix (A = Q R)") {
  ndarray A = stack3();
  linalg::QRResult qr = linalg::qr(A, "reduced");
  CHECK((qr.q.shape() == Shape{3, 2, 2}));
  for (int64_t k = 0; k < 3; ++k) {
    ndarray recon = matmul(qr.q[k].copy(), qr.r[k].copy());
    CHECK(allclose(recon, A[k].copy(), 1e-9, 1e-11, true));
  }
}

TEST_CASE("batched svd reconstructs each matrix (A = U S Vh)") {
  ndarray A = stack3();
  linalg::SVDResult s = linalg::svd(A, false);
  CHECK((s.s.shape() == Shape{3, 2}));
  for (int64_t k = 0; k < 3; ++k) {
    ndarray u = s.u[k].copy(), sv = s.s[k].copy(), vh = s.vh[k].copy();
    ndarray usv = matmul(multiply(u, sv.reshape({1, 2})), vh);
    CHECK(allclose(usv, A[k].copy(), 1e-8, 1e-10, true));
  }
}

TEST_CASE("batched det over two leading axes matches numpy") {
  ndarray B = arange(0.0, 36.0, 1.0).reshape({2, 2, 3, 3});
  CHECK((linalg::det(B).shape() == Shape{2, 2}));
  auto o = npt::oracle("B=np.arange(36.).reshape(2,2,3,3); a=np.linalg.det(B)");
  if (o) CHECK(allclose(linalg::det(B), *o, 1e-7, 1e-9, true));
}

TEST_CASE("2-D linalg still returns the single-matrix result") {
  ndarray M = arange(0.0, 4.0, 1.0).reshape({2, 2});
  CHECK((linalg::det(M).shape() == Shape{}));
  CHECK((linalg::inv(M).shape() == Shape{2, 2}));
}
