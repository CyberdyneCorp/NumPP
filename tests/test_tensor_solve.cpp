#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <vector>

using namespace numpp;

namespace {
// A symmetric tridiagonal (hence invertible) 4x4 matrix, row-major.
std::vector<double> tridiag4() {
  return {4, 1, 0, 0, 1, 4, 1, 0, 0, 1, 4, 1, 0, 0, 1, 4};
}
ndarray mk(const std::vector<double>& v, Shape shape) {
  ndarray a(shape, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
const char* PY_A = "M=np.array([4,1,0,0,1,4,1,0,0,1,4,1,0,0,1,4.]).reshape(2,2,2,2); ";
}  // namespace

TEST_CASE("tensorsolve matches numpy") {
  ndarray a = mk(tridiag4(), {2, 2, 2, 2});
  ndarray b = mk({1, 2, 3, 4}, {2, 2});
  ndarray x = linalg::tensorsolve(a, b);
  CHECK((x.shape() == Shape{2, 2}));
  auto o = npt::oracle(std::string(PY_A) +
                       "b=np.array([1,2,3,4.]).reshape(2,2); a=np.linalg.tensorsolve(M,b)");
  if (o) CHECK(allclose(x, *o, 1e-9, 1e-11, true));
}

TEST_CASE("tensorsolve solution reconstructs b") {
  ndarray a = mk(tridiag4(), {2, 2, 2, 2});
  ndarray b = mk({1, 2, 3, 4}, {2, 2});
  ndarray x = linalg::tensorsolve(a, b);
  // M @ x.ravel() == b.ravel() (as a 4x1 column to satisfy 2-D matmul)
  ndarray M = mk(tridiag4(), {4, 4});
  ndarray recon = matmul(M, x.reshape({4, 1}));
  CHECK(allclose(recon, b.reshape({4, 1}), 1e-9, 1e-11, true));
}

TEST_CASE("tensorinv matches numpy") {
  ndarray a = mk(tridiag4(), {2, 2, 2, 2});
  ndarray ia = linalg::tensorinv(a, 2);
  CHECK((ia.shape() == Shape{2, 2, 2, 2}));
  auto o = npt::oracle(std::string(PY_A) + "a=np.linalg.tensorinv(M,2)");
  if (o) CHECK(allclose(ia, *o, 1e-9, 1e-11, true));
}

TEST_CASE("tensorinv is the inverse under tensordot") {
  ndarray a = mk(tridiag4(), {2, 2, 2, 2});
  ndarray ia = linalg::tensorinv(a, 2);
  // reshape both to (4,4): ia @ a == I
  ndarray prod = matmul(ia.reshape({4, 4}), a.reshape({4, 4}));
  auto o = npt::oracle("a=np.eye(4)");
  if (o) CHECK(allclose(prod, *o, 1e-9, 1e-11, true));
}

TEST_CASE("tensorsolve with axes matches numpy") {
  // a shape (2,2,2,2); move axis 0 to the end before solving.
  ndarray a = mk(tridiag4(), {2, 2, 2, 2});
  ndarray b = mk({1, 2, 3, 4}, {2, 2});
  ndarray x = linalg::tensorsolve(a, b, {0});
  auto o = npt::oracle(std::string(PY_A) +
                       "b=np.array([1,2,3,4.]).reshape(2,2); a=np.linalg.tensorsolve(M,b,axes=(0,))");
  if (o) CHECK(allclose(x, *o, 1e-9, 1e-11, true));
}

TEST_CASE("tensorinv rejects ind <= 0") {
  ndarray a = mk(tridiag4(), {2, 2, 2, 2});
  CHECK_THROWS_AS(linalg::tensorinv(a, 0), value_error);
}
