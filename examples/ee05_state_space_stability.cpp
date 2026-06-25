#include "parity.hpp"  // ex::check / ex::check_scalar / ex::summary + numpp/numpp.hpp
#include <algorithm>
#include <complex>
#include <cstdio>
#include <vector>
using namespace numpp;

// Linear time-invariant (LTI) state-space model:  x_dot = A x.
// The system is asymptotically STABLE iff every eigenvalue of the state
// matrix A has a strictly negative real part (the poles live in the open
// left half-plane). Here A has a complex-conjugate pole pair (oscillation)
// plus one purely real pole, all with Re < 0, so the system is stable.
int main() {
  std::printf("== EE: State-Space Stability via Eigenvalues ==\n");

  // State matrix A (3x3). Top-left 2x2 block gives poles -1 +/- 5i
  // (damped oscillator); the third state has a real pole at -2.
  ndarray A = zeros({3, 3}, kFloat64);
  A.set_item<double>({0, 0}, -1.0); A.set_item<double>({0, 1}, -5.0);
  A.set_item<double>({1, 0},  5.0); A.set_item<double>({1, 1}, -1.0);
  A.set_item<double>({2, 2}, -2.0);

  // Trace = sum of eigenvalues; Determinant = product of eigenvalues.
  double tr  = trace(A).item<double>({});
  double det = linalg::det(A).item<double>({});
  ex::check_scalar("trace(A) = sum of poles", tr, -4.0);
  ex::check_scalar("det(A)   = product of poles", det, -52.0);

  // Eigenvalues (poles) as a complex array of length 3.
  ndarray ev = linalg::eigvals(A);

  // Sort by (real, imag) so the ordering matches numpy.sort_complex, which is
  // needed because eigvals() makes no ordering guarantee.
  std::vector<std::complex<double>> poles;
  for (int64_t i = 0; i < ev.size(); ++i)
    poles.push_back(ev.item<std::complex<double>>({i}));
  std::sort(poles.begin(), poles.end(),
            [](auto& l, auto& r) {
              return l.real() != r.real() ? l.real() < r.real()
                                          : l.imag() < r.imag();
            });
  ndarray ev_sorted = zeros({3}, kComplex128);
  for (int64_t i = 0; i < 3; ++i)
    ev_sorted.set_item<std::complex<double>>({i}, poles[i]);

  ex::check("eigvals(A) sorted (complex poles)", ev_sorted,
            "a = np.sort_complex(np.linalg.eigvals("
            "np.array([[-1,-5,0],[5,-1,0],[0,0,-2]], float)))");

  // Real and imaginary parts of the (sorted) poles.
  ex::check("Re(poles)", real(ev_sorted),
            "a = np.sort_complex(np.linalg.eigvals("
            "np.array([[-1,-5,0],[5,-1,0],[0,0,-2]], float))).real");
  ex::check("Im(poles)", imag(ev_sorted),
            "a = np.sort_complex(np.linalg.eigvals("
            "np.array([[-1,-5,0],[5,-1,0],[0,0,-2]], float))).imag");

  // |poles|: natural frequency of each mode (magnitude of the pole).
  ex::check("|poles| (natural frequencies)", absolute(ev_sorted),
            "a = np.abs(np.sort_complex(np.linalg.eigvals("
            "np.array([[-1,-5,0],[5,-1,0],[0,0,-2]], float))))");

  // Stability margin: the largest (least negative) real part. Stable iff < 0.
  double max_re = poles.back().real();  // sorted ascending by real part
  for (auto& p : poles) max_re = std::max(max_re, p.real());
  ex::check_scalar("spectral abscissa max Re(pole)", max_re, -1.0);

  bool stable = max_re < 0.0;
  ex::check_scalar("is_stable (all Re<0)", stable ? 1.0 : 0.0, 1.0);

  return ex::summary();
}
