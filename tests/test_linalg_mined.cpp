// Cases mined from numpy's own numpy/linalg/tests/test_linalg.py: the seed arrays
// (real/complex, single/double, square/non-square/Hermitian/size-0/1x1) and the
// stacked (generalized) variants, run through the live-NumPy oracle. Validates the
// deterministic linalg outputs (sign/order-ambiguous decompositions are checked by
// reconstruction or singular/eigenvalues).
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <complex>
#include <limits>
#include <vector>

using namespace numpp;
using cd = std::complex<double>;

namespace {
ndarray rmat(int64_t r, int64_t c, std::vector<double> v) {
  ndarray a(Shape{r, c}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
ndarray cmat(int64_t r, int64_t c, std::vector<cd> v) {
  ndarray a(Shape{r, c}, kComplex128, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<cd>()[i] = v[i];
  return a;
}
const double INF = std::numeric_limits<double>::infinity();
// numpy seed: cdouble square 2x2 from test_linalg.py ("cdouble" case scaled).
ndarray cA() { return cmat(2, 2, {cd(1, 2), cd(2, 3), cd(3, 4), cd(4, 5)}); }
ndarray cB() { return cmat(2, 1, {cd(2, 1), cd(1, 2)}); }
}  // namespace

// ---- complex square: solve / inv / det / slogdet (numpy SolveCases/InvCases/DetCases) ----
TEST_CASE("mined: complex solve vs numpy") {
  ndarray a = cA(), b = cB();
  auto o = npt::oracle(
      "A=np.array([[1+2j,2+3j],[3+4j,4+5j]]); b=np.array([[2+1j],[1+2j]]); a=np.linalg.solve(A,b)");
  if (o) CHECK(allclose(linalg::solve(a, b), *o, 1e-9, 1e-11, true));
}
TEST_CASE("mined: complex inv vs numpy") {
  auto o = npt::oracle("A=np.array([[1+2j,2+3j],[3+4j,4+5j]]); a=np.linalg.inv(A)");
  if (o) CHECK(allclose(linalg::inv(cA()), *o, 1e-9, 1e-11, true));
}
TEST_CASE("mined: complex det vs numpy") {
  auto o = npt::oracle("A=np.array([[1+2j,2+3j],[3+4j,4+5j]]); a=np.array(np.linalg.det(A))");
  if (o) CHECK(allclose(linalg::det(cA()), *o, 1e-9, 1e-11, true));
}
TEST_CASE("mined: complex slogdet vs numpy") {
  linalg::SignLogDet sd = linalg::slogdet(cA());
  auto os = npt::oracle("A=np.array([[1+2j,2+3j],[3+4j,4+5j]]); a=np.array(np.linalg.slogdet(A)[0])");
  auto ol = npt::oracle("A=np.array([[1+2j,2+3j],[3+4j,4+5j]]); a=np.array(np.linalg.slogdet(A)[1])");
  if (os) CHECK(allclose(sd.sign, *os, 1e-9, 1e-11, true));
  if (ol) CHECK(allclose(sd.logabsdet, *ol, 1e-9, 1e-11, true));
}
TEST_CASE("mined: complex matrix_power vs numpy") {
  auto o = npt::oracle("A=np.array([[1+2j,2+3j],[3+4j,4+5j]]); a=np.linalg.matrix_power(A,3)");
  if (o) CHECK(allclose(linalg::matrix_power(cA(), 3), *o, 1e-7, 1e-9, true));
}

// ---- complex non-square: svdvals / pinv / lstsq (numpy SVDCases/PinvCases/LstsqCases) ----
TEST_CASE("mined: complex non-square svdvals vs numpy") {
  ndarray a = cmat(3, 2, {cd(1, 1), cd(2, 2), cd(3, -3), cd(4, -9), cd(5, -4), cd(6, 8)});
  auto o = npt::oracle(
      "A=np.array([[1+1j,2+2j],[3-3j,4-9j],[5-4j,6+8j]]); a=np.linalg.svd(A,compute_uv=False)");
  if (o) CHECK(allclose(linalg::svdvals(a), *o, 1e-8, 1e-10, true));
}
TEST_CASE("mined: complex non-square pinv vs numpy") {
  ndarray a = cmat(3, 2, {cd(1, 1), cd(2, 2), cd(3, -3), cd(4, -9), cd(5, -4), cd(6, 8)});
  auto o = npt::oracle(
      "A=np.array([[1+1j,2+2j],[3-3j,4-9j],[5-4j,6+8j]]); a=np.linalg.pinv(A)");
  if (o) CHECK(allclose(linalg::pinv(a), *o, 1e-7, 1e-9, true));
}

// ---- Hermitian complex: eigvalsh (numpy TestEigvalshCases) ----
TEST_CASE("mined: hermitian complex eigvalsh vs numpy") {
  ndarray a = cmat(2, 2, {cd(1, 0), cd(2, 3), cd(2, -3), cd(1, 0)});
  auto o = npt::oracle("A=np.array([[1,2+3j],[2-3j,1]]); a=np.linalg.eigvalsh(A)");
  if (o) CHECK(allclose(linalg::eigvalsh(a), *o, 1e-9, 1e-11, true));
}

// ---- size-0 / empty (numpy tags={'size-0'}) ----
TEST_CASE("mined: inv of a 0x0 matrix vs numpy") {
  ndarray a(Shape{0, 0}, kFloat64, Order::C);
  auto o = npt::oracle("a=np.linalg.inv(np.empty((0,0)))");
  if (o) CHECK(allclose(linalg::inv(a), *o, 1e-9, 1e-11, true));
}
TEST_CASE("mined: det of a 0x0 matrix is 1.0 vs numpy") {
  ndarray a(Shape{0, 0}, kFloat64, Order::C);
  auto o = npt::oracle("a=np.array(np.linalg.det(np.empty((0,0))))");
  if (o) CHECK(allclose(linalg::det(a), *o, 1e-9, 1e-11, true));
}

// ---- 1x1 (numpy hmatrix_1x1 etc.) ----
TEST_CASE("mined: 1x1 inv / det vs numpy") {
  ndarray a = rmat(1, 1, {5.0});
  auto oi = npt::oracle("a=np.linalg.inv(np.array([[5.]]))");
  auto od = npt::oracle("a=np.array(np.linalg.det(np.array([[5.]])))");
  if (oi) CHECK(allclose(linalg::inv(a), *oi, 1e-12, 1e-12, true));
  if (od) CHECK(allclose(linalg::det(a), *od, 1e-12, 1e-12, true));
}

// ---- stacked complex (numpy generalized tile3) ----
TEST_CASE("mined: stacked complex det vs numpy") {
  // a = np.array([A, 2A, 3A]) for the cdouble 2x2 seed.
  ndarray a(Shape{3, 2, 2}, kComplex128, Order::C);
  cd base[4] = {cd(1, 2), cd(2, 3), cd(3, 4), cd(4, 5)};
  for (int k = 0; k < 3; ++k)
    for (int i = 0; i < 4; ++i) a.typed_data<cd>()[k * 4 + i] = base[i] * static_cast<double>(k + 1);
  auto o = npt::oracle(
      "A=np.array([[1+2j,2+3j],[3+4j,4+5j]]); S=np.array([A,2*A,3*A]); a=np.linalg.det(S)");
  if (o) CHECK(allclose(linalg::det(a), *o, 1e-7, 1e-9, true));
}

// ---- matrix norm ords (numpy TestNorm matrix cases) ----
TEST_CASE("mined: matrix norm ords vs numpy") {
  ndarray A = rmat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 10});  // non-singular
  const char* PYA = "A=np.array([[1,2,3],[4,5,6],[7,8,10.]]); ";
  auto check = [&](const ndarray& got, const std::string& expr) {
    auto o = npt::oracle(std::string(PYA) + "a=np.array(" + expr + ")");
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  check(linalg::norm(A, std::string("fro")), "np.linalg.norm(A,'fro')");
  check(linalg::norm(A, 1.0), "np.linalg.norm(A,1)");
  check(linalg::norm(A, -1.0), "np.linalg.norm(A,-1)");
  check(linalg::norm(A, 2.0), "np.linalg.norm(A,2)");
  check(linalg::norm(A, -2.0), "np.linalg.norm(A,-2)");
  check(linalg::norm(A, INF), "np.linalg.norm(A,np.inf)");
  check(linalg::norm(A, -INF), "np.linalg.norm(A,-np.inf)");
  check(linalg::norm(A, std::string("nuc")), "np.linalg.norm(A,'nuc')");
}

// ---- vector norm ords + axis (numpy TestNorm vector cases) ----
TEST_CASE("mined: vector norm ords vs numpy") {
  ndarray v = rmat(1, 5, {1, -2, 3, -4, 5}).reshape({5});
  const char* PYV = "v=np.array([1,-2,3,-4,5.]); ";
  auto check = [&](const ndarray& got, const std::string& expr) {
    auto o = npt::oracle(std::string(PYV) + "a=np.array(" + expr + ")");
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  check(linalg::norm(v, 1.0), "np.linalg.norm(v,1)");
  check(linalg::norm(v, 2.0), "np.linalg.norm(v,2)");
  check(linalg::norm(v, INF), "np.linalg.norm(v,np.inf)");
  check(linalg::norm(v, -INF), "np.linalg.norm(v,-np.inf)");
  check(linalg::norm(v, 0.0), "np.linalg.norm(v,0)");
  check(linalg::norm(v, 3.0), "np.linalg.norm(v,3)");
}

// ---- cond ords (numpy TestCond) ----
TEST_CASE("mined: cond ords vs numpy") {
  ndarray A = rmat(2, 2, {1, 0, 0, 4});  // numpy TestCond.test_basic_nonsvd matrix-ish
  const char* PYA = "A=np.array([[1,0],[0,4.]]); ";
  auto check = [&](const ndarray& got, const std::string& expr) {
    auto o = npt::oracle(std::string(PYA) + "a=np.array(" + expr + ")");
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  check(cond(A), "np.linalg.cond(A)");  // 2-norm default
}

// ---- SVD relative accuracy on a true-zero singular value (regression for #74) ----
TEST_CASE("mined: SVD resolves a true-zero singular value (regression #74)") {
  ndarray A = rmat(3, 3, {1, 2, 3, 2, 4, 6, 1, 0, 0});  // row2 = 2*row1 -> sigma_min ~ 0
  ndarray s = linalg::svdvals(A);
  // The Gram/A^T A approach left this at ~5.97e-8 (half precision); one-sided
  // Jacobi resolves it near machine zero.
  CHECK(s.item<double>({2}) < 1e-10);
}

// ---- matrix_rank (numpy TestMatrixRank) ----
TEST_CASE("mined: matrix_rank of a rank-deficient matrix vs numpy") {
  ndarray A = rmat(3, 3, {1, 2, 3, 2, 4, 6, 1, 0, 0});  // row2 = 2*row1 -> rank 2
  auto o = npt::oracle("A=np.array([[1,2,3],[2,4,6],[1,0,0.]]); a=np.array(np.linalg.matrix_rank(A))");
  if (o) CHECK(allclose(linalg::matrix_rank(A), *o, 0, 0, true));
}

// ---- dtype preservation (numpy TestSolve.test_types) ----
TEST_CASE("mined: solve preserves float32 dtype") {
  ndarray x(Shape{2, 2}, kFloat32, Order::C);
  float v[4] = {1, 0.5f, 0.5f, 1};
  for (int i = 0; i < 4; ++i) x.typed_data<float>()[i] = v[i];
  CHECK(linalg::solve(x, x).dtype() == kFloat32);
}
