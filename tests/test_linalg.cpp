#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s (no oracle)\n", label); return; }
  CHECK(allclose(got, *o, 1e-6, 1e-8, true));
}

// A well-conditioned 3x3 matrix mirrored on both sides.
static ndarray M() {
  ndarray m({3, 3}, kFloat64);
  double vals[9] = {4, 2, 1, 2, 5, 3, 1, 3, 6};  // symmetric positive-definite
  for (int i = 0; i < 9; ++i) m.set_item<double>({i / 3, i % 3}, vals[i]);
  return m;
}
static const char* Mpy = "M=np.array([[4.,2,1],[2,5,3],[1,3,6]])";

TEST_CASE("dot / vdot / inner / outer") {
  ndarray a = arange(0.0, 3.0, 1.0), b = arange(3.0, 6.0, 1.0);
  chk(dot(a, b), "a=np.array(np.dot(np.arange(3.),np.arange(3,6.)))", "dot 1d");
  chk(dot(M(), M()), std::string(Mpy) + ";a=M.dot(M)", "dot 2d");
  chk(outer(a, b), "a=np.outer(np.arange(3.),np.arange(3,6.))", "outer");
  chk(inner(M(), M()), std::string(Mpy) + ";a=np.inner(M,M)", "inner 2d");
  ndarray c0 = arange(1.0, 4.0, 1.0).astype(kComplex128);
  chk(vdot(add(c0, multiply(c0, scalar_like(0.0, kComplex128, true))), c0),
      "z=np.arange(1,4).astype(np.complex128);a=np.array(np.vdot(z,z))", "vdot");
}

TEST_CASE("trace and kron") {
  chk(trace(M()), std::string(Mpy) + ";a=np.array(np.trace(M))", "trace");
  ndarray a = arange(1.0, 5.0, 1.0).reshape({2, 2});
  ndarray b = arange(0.0, 4.0, 1.0).reshape({2, 2});
  chk(kron(a, b), "a=np.kron(np.arange(1,5.).reshape(2,2),np.arange(4.).reshape(2,2))", "kron");
}

TEST_CASE("solve, inv, det, slogdet") {
  ndarray bb({3}, kFloat64);
  bb.set_item<double>({0}, 1.0); bb.set_item<double>({1}, 2.0); bb.set_item<double>({2}, 3.0);
  ndarray x = linalg::solve(M(), bb);
  chk(x, std::string(Mpy) + ";a=np.linalg.solve(M,np.array([1.,2,3]))", "solve");
  // verify M @ x == b
  CHECK(allclose(matmul(M(), x.reshape({3, 1})).reshape({3}), bb, 1e-9, 1e-9));

  chk(linalg::inv(M()), std::string(Mpy) + ";a=np.linalg.inv(M)", "inv");
  chk(linalg::det(M()), std::string(Mpy) + ";a=np.array(np.linalg.det(M))", "det");

  auto sld = linalg::slogdet(M());
  chk(sld.sign, std::string(Mpy) + ";a=np.array(np.linalg.slogdet(M)[0])", "slogdet sign");
  chk(sld.logabsdet, std::string(Mpy) + ";a=np.array(np.linalg.slogdet(M)[1])", "slogdet logabs");
}

TEST_CASE("singular matrix raises") {
  ndarray s = zeros({2, 2}, kFloat64);  // singular
  CHECK_THROWS_AS(linalg::inv(s), linalg_error);
}

TEST_CASE("cholesky") {
  chk(linalg::cholesky(M()), std::string(Mpy) + ";a=np.linalg.cholesky(M)", "cholesky");
  // reconstruct A = L L^T
  ndarray L = linalg::cholesky(M());
  CHECK(allclose(matmul(L, L.transpose()), M(), 1e-9, 1e-9));
  ndarray notpd = zeros({2, 2}, kFloat64);
  notpd.set_item<double>({0, 0}, -1.0); notpd.set_item<double>({1, 1}, -1.0);
  CHECK_THROWS_AS(linalg::cholesky(notpd), linalg_error);
}

TEST_CASE("matrix_power") {
  ndarray a = arange(1.0, 5.0, 1.0).reshape({2, 2});
  chk(linalg::matrix_power(a, 3), "a=np.linalg.matrix_power(np.arange(1,5.).reshape(2,2),3)", "mpow3");
  chk(linalg::matrix_power(a, 0), "a=np.linalg.matrix_power(np.arange(1,5.).reshape(2,2),0)", "mpow0");
  chk(linalg::matrix_power(M(), -1), std::string(Mpy) + ";a=np.linalg.matrix_power(M,-1)", "mpow-1");
}

TEST_CASE("complex solve") {
  ndarray a({2, 2}, kComplex128);
  a.set_item<std::complex<double>>({0, 0}, {1, 1}); a.set_item<std::complex<double>>({0, 1}, {2, 0});
  a.set_item<std::complex<double>>({1, 0}, {3, -1}); a.set_item<std::complex<double>>({1, 1}, {1, 2});
  ndarray b({2}, kComplex128);
  b.set_item<std::complex<double>>({0}, {1, 0}); b.set_item<std::complex<double>>({1}, {0, 1});
  ndarray x = linalg::solve(a, b);
  CHECK(allclose(matmul(a, x.reshape({2, 1})).reshape({2}), b, 1e-9, 1e-9));
}
