// Deeper cases mined from numpy's numpy/linalg/tests/test_linalg.py: cond with all
// norms (TestCond.test_basic_nonsvd / test_singular), lstsq, multi_dot, and eigh
// reconstruction for a Hermitian complex matrix.
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
}  // namespace

TEST_CASE("mined: cond with every norm vs numpy (TestCond.test_basic_nonsvd)") {
  ndarray A = rmat(3, 3, {1, 0, 1, 0, -2, 0, 0, 0, 3});
  const char* PYA = "A=np.array([[1.,0,1],[0,-2,0],[0,0,3]]); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYA) + "a=np.array(" + e + ")");
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  chk(cond(A, INF), "np.linalg.cond(A,np.inf)");
  chk(cond(A, -INF), "np.linalg.cond(A,-np.inf)");
  chk(cond(A, 1.0), "np.linalg.cond(A,1)");
  chk(cond(A, -1.0), "np.linalg.cond(A,-1)");
  chk(cond(A, std::string("fro")), "np.linalg.cond(A,'fro')");
  chk(cond(A, 2.0), "np.linalg.cond(A,2)");
}

TEST_CASE("mined: cond 2 / -2 vs numpy") {
  ndarray A = rmat(2, 2, {1, 0, 0, 4});
  auto o2 = npt::oracle("a=np.array(np.linalg.cond(np.array([[1.,0],[0,4]]),2))");
  auto om = npt::oracle("a=np.array(np.linalg.cond(np.array([[1.,0],[0,4]]),-2))");
  if (o2) CHECK(allclose(cond(A, 2.0), *o2, 1e-12, 1e-12, true));
  if (om) CHECK(allclose(cond(A, -2.0), *om, 1e-12, 1e-12, true));
}

TEST_CASE("mined: cond of a singular matrix is infinite for positive norms") {
  ndarray ones = rmat(2, 2, {1, 1, 1, 1});  // rank 1 -> singular
  CHECK(cond(ones, 1.0).item<double>({}) > 1e15);
  CHECK(cond(ones, 2.0).item<double>({}) > 1e15);
  CHECK(cond(ones, std::string("fro")).item<double>({}) > 1e15);
  ndarray zero = rmat(2, 2, {0, 0, 0, 0});
  CHECK(cond(zero, 2.0).item<double>({}) > 1e15);
  // Negative norms must not throw.
  (void)cond(ones, -1.0);
  (void)cond(ones, -2.0);
}

TEST_CASE("mined: lstsq overdetermined vs numpy (LstsqCases)") {
  ndarray A = rmat(3, 2, {1, 1, 1, 2, 1, 3});
  ndarray b = rmat(3, 1, {6, 5, 7}).reshape({3});
  linalg::LstsqResult r = linalg::lstsq(A, b);
  auto ox = npt::oracle(
      "A=np.array([[1,1.],[1,2],[1,3]]); b=np.array([6,5,7.]); a=np.linalg.lstsq(A,b,rcond=None)[0]");
  if (ox) CHECK(allclose(r.solution, *ox, 1e-7, 1e-9, true));
  CHECK(r.rank.item<int64_t>({}) == 2);
}

TEST_CASE("mined: multi_dot vs numpy") {
  ndarray A = arange(0.0, 6.0, 1.0).reshape({2, 3});
  ndarray B = arange(0.0, 12.0, 1.0).reshape({3, 4});
  ndarray C = arange(0.0, 8.0, 1.0).reshape({4, 2});
  ndarray got = multi_dot({A, B, C});
  auto o = npt::oracle(
      "A=np.arange(6.).reshape(2,3); B=np.arange(12.).reshape(3,4); C=np.arange(8.).reshape(4,2); "
      "a=np.linalg.multi_dot([A,B,C])");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}

TEST_CASE("mined: eigh of a Hermitian complex matrix vs numpy (TestEighCases)") {
  ndarray A = cmat(2, 2, {cd(2, 0), cd(1, 1), cd(1, -1), cd(3, 0)});  // Hermitian
  linalg::EighResult e = linalg::eigh(A);
  // eigenvalues ascending, unique -> compare directly.
  auto ow = npt::oracle("A=np.array([[2,1+1j],[1-1j,3]]); a=np.linalg.eigvalsh(A)");
  if (ow) CHECK(allclose(e.eigenvalues, *ow, 1e-9, 1e-11, true));
  // reconstruction: V @ diag(w) @ V^H == A
  ndarray w = e.eigenvalues.astype(kComplex128);
  ndarray D(Shape{2, 2}, kComplex128, Order::C);
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
    D.set_item<cd>({i, j}, i == j ? cd(e.eigenvalues.item<double>({i}), 0) : cd(0, 0));
  ndarray Vh = conj(e.eigenvectors).transpose();
  ndarray recon = matmul(matmul(e.eigenvectors, D), Vh);
  CHECK(allclose(recon, A, 1e-9, 1e-11, true));
}
