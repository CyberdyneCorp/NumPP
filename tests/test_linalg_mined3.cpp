// Final batch mined from numpy's numpy/linalg/tests/test_linalg.py:
// MatrixPowerCases (zero/one/two/negative/large), TestLstsq (rank + residuals),
// TestEig (non-symmetric reconstruction), TestNorm (axis), TestQR (modes).
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

// ---- MatrixPowerCases: 0 / 1 / 2 / negative / large ----
TEST_CASE("mined: matrix_power exponents vs numpy") {
  ndarray A = rmat(2, 2, {1, 2, 3, 4});
  const char* PYA = "A=np.array([[1,2],[3,4.]]); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYA) + "a=" + e);
    if (o) CHECK(allclose(got, *o, 1e-7, 1e-9, true));
  };
  chk(linalg::matrix_power(A, 0), "np.linalg.matrix_power(A,0)");   // identity
  chk(linalg::matrix_power(A, 1), "np.linalg.matrix_power(A,1)");
  chk(linalg::matrix_power(A, 2), "np.linalg.matrix_power(A,2)");
  chk(linalg::matrix_power(A, -1), "np.linalg.matrix_power(A,-1)");  // inverse
  chk(linalg::matrix_power(A, -3), "np.linalg.matrix_power(A,-3)");
  chk(linalg::matrix_power(A, 7), "np.linalg.matrix_power(A,7)");
}

TEST_CASE("mined: matrix_power(0) is identity and preserves dtype") {
  ndarray Af = rmat(2, 2, {1, 2, 3, 4}).astype(kFloat32);
  CHECK(linalg::matrix_power(Af, 0).dtype() == kFloat32);
  ndarray Ac = cmat(2, 2, {cd(1, 1), cd(2, 0), cd(0, 1), cd(3, 0)});
  CHECK(linalg::matrix_power(Ac, 0).dtype() == kComplex128);
}

// ---- TestLstsq: overdetermined full-rank (solution, rank, residuals) ----
TEST_CASE("mined: lstsq overdetermined solution/rank/residuals vs numpy") {
  ndarray A = rmat(4, 2, {1, 1, 1, 2, 1, 3, 1, 4});
  ndarray b = rmat(4, 1, {6, 5, 7, 10}).reshape({4});
  linalg::LstsqResult r = linalg::lstsq(A, b);
  auto ox = npt::oracle(
      "A=np.array([[1,1.],[1,2],[1,3],[1,4]]); b=np.array([6,5,7,10.]); "
      "a=np.linalg.lstsq(A,b,rcond=None)[0]");
  if (ox) CHECK(allclose(r.solution, *ox, 1e-7, 1e-9, true));
  CHECK(r.rank.item<int64_t>({}) == 2);
  auto ores = npt::oracle(
      "A=np.array([[1,1.],[1,2],[1,3],[1,4]]); b=np.array([6,5,7,10.]); "
      "a=np.linalg.lstsq(A,b,rcond=None)[1]");
  if (ores) CHECK(allclose(r.residuals, *ores, 1e-7, 1e-9, true));
}

// ---- TestLstsq.test_rcond: rank-deficient rank matches numpy(rcond=None) ----
TEST_CASE("mined: lstsq rank-deficient rank vs numpy(rcond=None)") {
  ndarray a = rmat(6, 4, {0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2,
                          2, 1, 0, 3, 0, 0, 4, 0});
  ndarray b = rmat(6, 1, {1, 0, 0, 0, 0, 0}).reshape({6});
  linalg::LstsqResult r = linalg::lstsq(a, b);
  auto o = npt::oracle(
      "a=np.array([[0.,1,0,1,2,0],[0,2,0,0,1,0],[1,0,1,0,0,4],[0,0,0,2,3,0]]).T; "
      "b=np.array([1.,0,0,0,0,0]); a=np.array(np.linalg.lstsq(a,b,rcond=None)[2])");
  if (o) CHECK(r.rank.item<int64_t>({}) == o->astype(kInt64).item<int64_t>({}));
}

// ---- TestEig: non-symmetric real matrix with complex eigenvalues (reconstruct) ----
TEST_CASE("mined: eig of a non-symmetric real matrix reconstructs A") {
  ndarray A = rmat(2, 2, {2, -1, 1, 2});  // eigenvalues 2 +/- i
  linalg::EigResult e = linalg::eig(A);
  ndarray V = e.eigenvectors;                          // complex
  ndarray D(Shape{2, 2}, kComplex128, Order::C);
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
    D.set_item<cd>({i, j}, i == j ? e.eigenvalues.item<cd>({i}) : cd(0, 0));
  ndarray recon = matmul(matmul(V, D), linalg::inv(V));
  CHECK(allclose(recon, A.astype(kComplex128), 1e-9, 1e-11, true));
}

TEST_CASE("mined: eig of a complex matrix reconstructs A") {
  ndarray A = cmat(2, 2, {cd(1, 2), cd(2, 0), cd(0, 1), cd(3, -1)});
  linalg::EigResult e = linalg::eig(A);
  ndarray D(Shape{2, 2}, kComplex128, Order::C);
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
    D.set_item<cd>({i, j}, i == j ? e.eigenvalues.item<cd>({i}) : cd(0, 0));
  ndarray recon = matmul(matmul(e.eigenvectors, D), linalg::inv(e.eigenvectors));
  CHECK(allclose(recon, A, 1e-9, 1e-11, true));
}

// ---- TestNorm: vector_norm with axis (array-API) ----
TEST_CASE("mined: vector_norm with axis vs numpy") {
  ndarray A = rmat(2, 3, {1, 2, 3, 4, 5, 6});
  const char* PYA = "A=np.array([[1,2,3],[4,5,6.]]); ";
  auto chk = [&](const ndarray& got, const std::string& e) {
    auto o = npt::oracle(std::string(PYA) + "a=" + e);
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  chk(linalg::vector_norm(A, int64_t{0}, 2.0), "np.linalg.norm(A,ord=2,axis=0)");
  chk(linalg::vector_norm(A, int64_t{1}, 2.0), "np.linalg.norm(A,ord=2,axis=1)");
  chk(linalg::vector_norm(A, int64_t{0}, 1.0), "np.linalg.norm(A,ord=1,axis=0)");
  chk(linalg::vector_norm(A, int64_t{1}, INF), "np.linalg.norm(A,ord=np.inf,axis=1)");
}

// ---- TestQR: reduced vs complete modes (shapes + reconstruction) ----
TEST_CASE("mined: qr reduced and complete reconstruct A (tall)") {
  ndarray A = rmat(3, 2, {1, 2, 3, 4, 5, 7});
  linalg::QRResult qr_r = linalg::qr(A, "reduced");
  CHECK((qr_r.q.shape() == Shape{3, 2}));
  CHECK((qr_r.r.shape() == Shape{2, 2}));
  CHECK(allclose(matmul(qr_r.q, qr_r.r), A, 1e-9, 1e-11, true));
  linalg::QRResult qr_c = linalg::qr(A, "complete");
  CHECK((qr_c.q.shape() == Shape{3, 3}));
  CHECK((qr_c.r.shape() == Shape{3, 2}));
  CHECK(allclose(matmul(qr_c.q, qr_c.r), A, 1e-9, 1e-11, true));
}

// ---- DetCases: singular determinant / slogdet ----
TEST_CASE("mined: det and slogdet of a singular matrix vs numpy") {
  ndarray A = rmat(2, 2, {1, 2, 2, 4});  // singular
  auto od = npt::oracle("a=np.array(np.linalg.det(np.array([[1,2],[2,4.]])))");
  if (od) CHECK(allclose(linalg::det(A), *od, 1e-12, 1e-12, true));
  linalg::SignLogDet sd = linalg::slogdet(A);
  CHECK(sd.sign.item<double>({}) == 0.0);
  CHECK(sd.logabsdet.item<double>({}) == -INF);
}
