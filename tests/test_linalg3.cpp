#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s (no oracle)\n", label); return; }
  CHECK(allclose(got, *o, 1e-6, 1e-8, true));
}
static ndarray rect() {  // 4x3
  ndarray m({4, 3}, kFloat64);
  double v[12] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 0, 2};
  for (int i = 0; i < 12; ++i) m.set_item<double>({i / 3, i % 3}, v[i]);
  return m;
}
static const char* Rpy = "R=np.array([[1.,2,3],[4,5,6],[7,8,10],[1,0,2]])";
static ndarray wide() {  // 3x4 = rect().T-ish
  ndarray m({3, 4}, kFloat64);
  double v[12] = {1, 4, 7, 1, 2, 5, 8, 0, 3, 6, 10, 2};
  for (int i = 0; i < 12; ++i) m.set_item<double>({i / 4, i % 4}, v[i]);
  return m;
}
static const char* Wpy = "W=np.array([[1.,4,7,1],[2,5,8,0],[3,6,10,2]])";

static ndarray diagS(const ndarray& s, int64_t m, int64_t n) {
  ndarray d = zeros({m, n}, s.dtype());
  int64_t k = s.shape()[0];
  for (int64_t i = 0; i < k; ++i) d.set_item<double>({i, i}, s.astype(kFloat64).item<double>({i}));
  return d;
}

TEST_CASE("svd singular values and reconstruction (tall)") {
  auto r = linalg::svd(rect(), /*full_matrices=*/false);
  CHECK(r.u.shape() == Shape({4, 3}));
  CHECK(r.s.shape() == Shape({3}));
  CHECK(r.vh.shape() == Shape({3, 3}));
  chk(r.s, std::string(Rpy) + ";a=np.linalg.svd(R,compute_uv=False)", "svdvals tall");
  // reconstruct U @ diag(S) @ Vh
  ndarray recon = matmul(r.u, matmul(diagS(r.s, 3, 3), r.vh));
  CHECK(allclose(recon, rect(), 1e-8, 1e-8));
}

TEST_CASE("svd full_matrices shapes and reconstruction (wide)") {
  auto r = linalg::svd(wide(), true);
  CHECK(r.u.shape() == Shape({3, 3}));
  CHECK(r.vh.shape() == Shape({4, 4}));
  ndarray recon = matmul(r.u, matmul(diagS(r.s, 3, 4), r.vh));
  CHECK(allclose(recon, wide(), 1e-8, 1e-8));
  chk(r.s, std::string(Wpy) + ";a=np.linalg.svd(W,compute_uv=False)", "svdvals wide");
}

TEST_CASE("pinv matches numpy") {
  chk(linalg::pinv(rect()), std::string(Rpy) + ";a=np.linalg.pinv(R)", "pinv tall");
  chk(linalg::pinv(wide()), std::string(Wpy) + ";a=np.linalg.pinv(W)", "pinv wide");
  // A @ A+ @ A == A
  ndarray p = linalg::pinv(rect());
  CHECK(allclose(matmul(rect(), matmul(p, rect())), rect(), 1e-7, 1e-7));
}

TEST_CASE("matrix_rank matches numpy") {
  chk(linalg::matrix_rank(rect()).astype(kFloat64),
      std::string(Rpy) + ";a=np.array(float(np.linalg.matrix_rank(R)))", "rank full");
  ndarray sing = zeros({3, 3}, kFloat64);
  sing.set_item<double>({0, 0}, 1.0); sing.set_item<double>({1, 1}, 2.0);  // rank 2
  CHECK(linalg::matrix_rank(sing).item<int64_t>({}) == 2);
}

TEST_CASE("lstsq solution matches numpy (overdetermined full rank)") {
  ndarray b({4}, kFloat64);
  double bv[4] = {1, 2, 3, 4};
  for (int i = 0; i < 4; ++i) b.set_item<double>({i}, bv[i]);
  auto r = linalg::lstsq(rect(), b);
  chk(r.solution, std::string(Rpy) + ";b=np.array([1.,2,3,4]);a=np.linalg.lstsq(R,b,rcond=-1)[0]", "lstsq x");
  CHECK(r.rank.item<int64_t>({}) == 3);
}

TEST_CASE("svd-based matrix norms vs numpy") {
  chk(linalg::norm(rect(), 2.0), std::string(Rpy) + ";a=np.array(np.linalg.norm(R,2))", "spectral");
  chk(linalg::norm(rect(), std::string("nuc")), std::string(Rpy) + ";a=np.array(np.linalg.norm(R,'nuc'))", "nuclear");
}

TEST_CASE("complex svd reconstruction") {
  ndarray a({2, 2}, kComplex128);
  a.set_item<std::complex<double>>({0, 0}, {1, 1}); a.set_item<std::complex<double>>({0, 1}, {2, -1});
  a.set_item<std::complex<double>>({1, 0}, {0, 3}); a.set_item<std::complex<double>>({1, 1}, {4, 0});
  auto r = linalg::svd(a, false);
  ndarray recon = matmul(r.u, matmul(diagS(r.s, 2, 2).astype(kComplex128), r.vh));
  CHECK(allclose(recon, a, 1e-8, 1e-8));
}
