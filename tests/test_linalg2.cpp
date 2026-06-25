#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s (no oracle)\n", label); return; }
  CHECK(allclose(got, *o, 1e-6, 1e-8, true));
}
static ndarray sym() {
  ndarray m({3, 3}, kFloat64);
  double v[9] = {4, 2, 1, 2, 5, 3, 1, 3, 6};
  for (int i = 0; i < 9; ++i) m.set_item<double>({i / 3, i % 3}, v[i]);
  return m;
}
static const char* Spy = "S=np.array([[4.,2,1],[2,5,3],[1,3,6]])";
static ndarray rect() {  // 4x3 general
  ndarray m({4, 3}, kFloat64);
  double v[12] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 0, 2};
  for (int i = 0; i < 12; ++i) m.set_item<double>({i / 3, i % 3}, v[i]);
  return m;
}

TEST_CASE("qr reduced: reconstruction and orthonormality") {
  auto qr = linalg::qr(rect());
  CHECK(qr.q.shape() == Shape({4, 3}));
  CHECK(qr.r.shape() == Shape({3, 3}));
  CHECK(allclose(matmul(qr.q, qr.r), rect(), 1e-9, 1e-9));            // A = Q R
  CHECK(allclose(matmul(qr.q.transpose(), qr.q), eye(3), 1e-9, 1e-9)); // Q^T Q = I
}

TEST_CASE("qr complete shapes") {
  auto qr = linalg::qr(rect(), "complete");
  CHECK(qr.q.shape() == Shape({4, 4}));
  CHECK(qr.r.shape() == Shape({4, 3}));
  CHECK(allclose(matmul(qr.q, qr.r), rect(), 1e-9, 1e-9));
}

TEST_CASE("complex qr reconstruction") {
  ndarray a({2, 2}, kComplex128);
  a.set_item<std::complex<double>>({0, 0}, {1, 1}); a.set_item<std::complex<double>>({0, 1}, {2, -1});
  a.set_item<std::complex<double>>({1, 0}, {3, 0}); a.set_item<std::complex<double>>({1, 1}, {0, 2});
  auto qr = linalg::qr(a);
  CHECK(allclose(matmul(qr.q, qr.r), a, 1e-9, 1e-9));
}

TEST_CASE("eigvalsh matches numpy (ascending)") {
  chk(linalg::eigvalsh(sym()), std::string(Spy) + ";a=np.linalg.eigvalsh(S)", "eigvalsh");
}

TEST_CASE("eigh reconstruction A V = V diag(w)") {
  auto e = linalg::eigh(sym());
  int n = 3;
  ndarray lhs = matmul(sym(), e.eigenvectors);
  ndarray rhs = multiply(e.eigenvectors, e.eigenvalues.reshape({1, n}));
  CHECK(allclose(lhs, rhs, 1e-8, 1e-8));
  // eigenvalues agree with eigvalsh
  CHECK(allclose(e.eigenvalues, linalg::eigvalsh(sym()), 1e-9, 1e-9));
}

TEST_CASE("complex hermitian eigh") {
  ndarray a({2, 2}, kComplex128);
  a.set_item<std::complex<double>>({0, 0}, {2, 0}); a.set_item<std::complex<double>>({0, 1}, {0, -1});
  a.set_item<std::complex<double>>({1, 0}, {0, 1}); a.set_item<std::complex<double>>({1, 1}, {2, 0});
  chk(linalg::eigvalsh(a), "H=np.array([[2,-1j],[1j,2]]);a=np.linalg.eigvalsh(H)", "herm eigvalsh");
  auto e = linalg::eigh(a);
  CHECK(allclose(matmul(a, e.eigenvectors),
                 multiply(e.eigenvectors, e.eigenvalues.astype(kComplex128).reshape({1, 2})), 1e-8, 1e-8));
}

TEST_CASE("vector norms vs numpy") {
  ndarray v = arange(-3.0, 4.0, 1.0);  // [-3..3]
  const char* vp = "v=np.arange(-3,4.0)";
  chk(linalg::norm(v), std::string(vp) + ";a=np.array(np.linalg.norm(v))", "norm2");
  chk(linalg::norm(v, 1.0), std::string(vp) + ";a=np.array(np.linalg.norm(v,1))", "norm1");
  chk(linalg::norm(v, std::numeric_limits<double>::infinity()),
      std::string(vp) + ";a=np.array(np.linalg.norm(v,np.inf))", "norminf");
  chk(linalg::norm(v, 0.0), std::string(vp) + ";a=np.array(np.linalg.norm(v,0))", "norm0");
  chk(linalg::norm(v, 3.0), std::string(vp) + ";a=np.array(np.linalg.norm(v,3))", "norm3");
}

TEST_CASE("matrix norms vs numpy") {
  chk(linalg::norm(sym(), std::string("fro")), std::string(Spy) + ";a=np.array(np.linalg.norm(S,'fro'))", "fro");
  chk(linalg::norm(sym(), 1.0), std::string(Spy) + ";a=np.array(np.linalg.norm(S,1))", "mnorm1");
  chk(linalg::norm(sym(), std::numeric_limits<double>::infinity()),
      std::string(Spy) + ";a=np.array(np.linalg.norm(S,np.inf))", "mnorminf");
}
