#include <algorithm>
#include <complex>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

// Compare two complex eigenvalue arrays as multisets (sorted by (re, im)).
static bool eigvals_match(const ndarray& got, const ndarray& want, double tol) {
  ndarray g = got.astype(kComplex128).copy(), w = want.astype(kComplex128).copy();
  if (g.shape() != w.shape()) return false;
  int64_t n = g.size();
  std::vector<std::complex<double>> gv(n), wv(n);
  for (int64_t i = 0; i < n; ++i) { gv[i] = g.item<std::complex<double>>({i}); wv[i] = w.item<std::complex<double>>({i}); }
  auto cmp = [](std::complex<double> a, std::complex<double> b) {
    if (std::abs(a.real() - b.real()) > 1e-9) return a.real() < b.real();
    return a.imag() < b.imag();
  };
  std::sort(gv.begin(), gv.end(), cmp);
  std::sort(wv.begin(), wv.end(), cmp);
  for (int64_t i = 0; i < n; ++i) if (std::abs(gv[i] - wv[i]) > tol) return false;
  return true;
}

static ndarray mat(const std::vector<double>& v, int64_t n) {
  ndarray m({n, n}, kFloat64);
  for (int64_t i = 0; i < n * n; ++i) m.set_item<double>({i / n, i % n}, v[i]);
  return m;
}

TEST_CASE("eigvals real matrix with real eigenvalues") {
  ndarray a = mat({2, 0, 0, 0, 3, 0, 0, 0, 5}, 3);  // diagonal -> 2,3,5
  ndarray w = linalg::eigvals(a);
  CHECK(w.dtype() == kFloat64);
  auto o = npt::oracle("a=np.linalg.eigvals(np.diag([2.,3,5]))");
  if (o) CHECK(eigvals_match(w, *o, 1e-7));
}

TEST_CASE("eigvals real matrix with complex-conjugate pair") {
  ndarray a = mat({0, -1, 1, 0}, 2);  // eigenvalues +/- i
  ndarray w = linalg::eigvals(a);
  CHECK(w.dtype() == kComplex128);
  auto o = npt::oracle("a=np.linalg.eigvals(np.array([[0.,-1],[1,0]]))");
  if (o) CHECK(eigvals_match(w, *o, 1e-7));
}

TEST_CASE("eigvals general nonsymmetric vs numpy") {
  ndarray a = mat({1, 2, 3, 0, 4, 5, 0, 0, 6}, 3);  // upper-tri -> 1,4,6
  auto o = npt::oracle("a=np.linalg.eigvals(np.array([[1.,2,3],[0,4,5],[0,0,6]]))");
  if (o) CHECK(eigvals_match(linalg::eigvals(a), *o, 1e-6));
  ndarray g = mat({4, 1, 2, 1, 5, 3, 2, 3, 6}, 3);  // symmetric -> real
  auto o2 = npt::oracle("a=np.linalg.eigvals(np.array([[4.,1,2],[1,5,3],[2,3,6]]))");
  if (o2) CHECK(eigvals_match(linalg::eigvals(g), *o2, 1e-6));
}

TEST_CASE("eig reconstruction A v = lambda v (real eigenvalues)") {
  ndarray a = mat({4, 1, 2, 1, 5, 3, 2, 3, 6}, 3);  // symmetric
  auto e = linalg::eig(a);
  CHECK(e.eigenvalues.dtype() == kFloat64);
  // A @ V == V @ diag(w)
  ndarray lhs = matmul(a, e.eigenvectors);
  ndarray rhs = multiply(e.eigenvectors, e.eigenvalues.reshape({1, 3}));
  CHECK(allclose(lhs, rhs, 1e-7, 1e-7));
}

TEST_CASE("eig reconstruction with complex eigenvalues") {
  ndarray a = mat({0, -1, 1, 0}, 2);  // +/- i
  auto e = linalg::eig(a);
  CHECK(e.eigenvalues.dtype() == kComplex128);
  ndarray lhs = matmul(a.astype(kComplex128), e.eigenvectors);
  ndarray rhs = multiply(e.eigenvectors, e.eigenvalues.reshape({1, 2}));
  CHECK(allclose(lhs, rhs, 1e-7, 1e-7));
}

TEST_CASE("eig of complex matrix") {
  ndarray a({2, 2}, kComplex128);
  a.set_item<std::complex<double>>({0, 0}, {1, 1}); a.set_item<std::complex<double>>({0, 1}, {2, 0});
  a.set_item<std::complex<double>>({1, 0}, {0, 1}); a.set_item<std::complex<double>>({1, 1}, {3, -1});
  auto e = linalg::eig(a);
  ndarray lhs = matmul(a, e.eigenvectors);
  ndarray rhs = multiply(e.eigenvectors, e.eigenvalues.reshape({1, 2}));
  CHECK(allclose(lhs, rhs, 1e-7, 1e-7));
}
