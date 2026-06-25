// EE: Laplace transforms & transfer functions.
//
// A linear time-invariant system has a transfer function H(s) = N(s)/D(s), the
// Laplace-domain ratio of output to input. The roots of D(s) are the system's
// POLES (they govern stability and natural response) and the roots of N(s) are
// its ZEROS. Finding roots of a polynomial is an eigenvalue problem: the roots
// of a monic polynomial are the eigenvalues of its companion matrix — exactly
// how numpy.roots works internally. We compute poles/zeros that way and check
// the DC gain H(0) and the frequency response H(jw).
#include "parity.hpp"
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
using namespace numpp;

// Roots of a polynomial given by coefficients c (highest power first), via the
// eigenvalues of the companion matrix — the same construction numpy.roots uses.
static ndarray poly_roots(const std::vector<double>& c) {
  int n = static_cast<int>(c.size()) - 1;            // degree
  ndarray A = zeros({n, n}, kFloat64);
  for (int j = 0; j < n; ++j) A.set_item<double>({0, j}, -c[j + 1] / c[0]);  // first row
  for (int i = 1; i < n; ++i) A.set_item<double>({i, i - 1}, 1.0);           // sub-diagonal
  return sort(linalg::eigvals(A));                   // lexicographic sort for stable compare
}
// Horner evaluation of a real polynomial at complex points s.
static ndarray polyval(const std::vector<double>& c, const ndarray& s) {
  ndarray r = full(s.shape(), c[0], kComplex128);
  for (size_t k = 1; k < c.size(); ++k) r = r * s + c[k];
  return r;
}

int main() {
  std::printf("== EE: Laplace transfer function H(s) = (s+2)/(s^2+3s+2) ==\n");
  std::vector<double> num = {1.0, 2.0};        // N(s) = s + 2          -> zero at -2
  std::vector<double> den = {1.0, 3.0, 2.0};   // D(s) = s^2 + 3s + 2   -> poles at -1, -2

  ex::check("poles = roots(denominator)", poly_roots(den), "a=np.sort_complex(np.roots([1.,3,2]))");
  ex::check("zeros = roots(numerator)", poly_roots(num), "a=np.sort_complex(np.roots([1.,2]))");

  // DC gain H(0) = N(0)/D(0) = 2/2 = 1.
  ex::check_scalar("DC gain H(0)", num.back() / den.back(),
                   ex::numpy("a=np.array([np.polyval([1.,2],0)/np.polyval([1.,3,2],0)])").item<double>({0}));

  // Frequency response H(jw) over a grid of angular frequencies.
  ndarray w = linspace(0.1, 50.0, 24);
  ndarray s = zeros({w.size()}, kComplex128);
  for (int64_t i = 0; i < w.size(); ++i) s.set_item<std::complex<double>>({i}, std::complex<double>(0.0, w.item<double>({i})));
  ndarray H = divide(polyval(num, s), polyval(den, s));
  ex::check("transfer function H(jw)", H,
            "w=np.linspace(0.1,50,24); s=1j*w; a=np.polyval([1.,2],s)/np.polyval([1.,3,2],s)");
  ex::check("magnitude |H(jw)|", absolute(H),
            "w=np.linspace(0.1,50,24); s=1j*w; a=np.abs(np.polyval([1.,2],s)/np.polyval([1.,3,2],s))");

  return ex::summary();
}
