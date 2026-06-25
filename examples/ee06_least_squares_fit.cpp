// Least-squares polynomial regression: fit y ~ c0 + c1*x + c2*x^2.
//
// EE/ML concept: many sensor-calibration and curve-fitting problems are
// overdetermined linear systems A*c = y with more equations (data points)
// than unknowns (coefficients). The minimum-residual solution is found by
// solving the normal equations implicitly via QR/SVD inside linalg::lstsq.
// We build the Vandermonde design matrix A (column j = x^j) and let lstsq
// return the coefficients, the SVD singular values, and the effective rank.
#include "parity.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
using namespace numpp;

int main() {
  std::printf("== EE06: Least-Squares Quadratic Fit (Vandermonde + lstsq) ==\n");

  // Sample points x = 0, 0.5, ..., 4.5 (10 points). Underlying noisy-ish data.
  const int m = 10;
  ndarray x = linspace(0.0, 4.5, m);  // m sample locations

  // "Measured" outputs: a quadratic with a small deterministic perturbation so
  // the system is genuinely overdetermined (no exact 3-coefficient fit).
  ndarray y = zeros({m}, kFloat64);
  for (int i = 0; i < m; ++i) {
    double xi = x.item<double>({i});
    double yi = 2.0 - 1.5 * xi + 0.75 * xi * xi + 0.1 * std::sin(3.0 * xi);
    y.set_item<double>({i}, yi);
  }

  // Vandermonde design matrix A (m x 3): columns [1, x, x^2].
  const int deg = 3;  // number of coefficients (quadratic => 3)
  ndarray A = zeros({m, deg}, kFloat64);
  for (int i = 0; i < m; ++i) {
    double xi = x.item<double>({i});
    A.set_item<double>({i, 0}, 1.0);
    A.set_item<double>({i, 1}, xi);
    A.set_item<double>({i, 2}, xi * xi);
  }

  // Solve the least-squares problem: minimize ||A c - y||_2.
  auto fit = linalg::lstsq(A, y);
  ndarray c = fit.solution;          // 3 coefficients [c0, c1, c2]
  ndarray sv = fit.singular_values;  // singular values of A (length 3)

  // Residual vector r = A c - y, and its Euclidean norm (fit quality).
  ndarray resid = dot(A, c) - y;
  double resid_norm = linalg::norm(resid).item<double>({});

  // The fitted coefficients reproduce the data closely (within the perturbation).
  std::printf("  coefficients: c0=%.6f c1=%.6f c2=%.6f\n",
              c.item<double>({0}), c.item<double>({1}), c.item<double>({2}));
  std::printf("  residual 2-norm = %.6e\n", resid_norm);

  // --- Verify against numpy.linalg.lstsq with the SAME Vandermonde matrix. ---
  // Reconstruct identical x/y/A in NumPy so the oracle is bit-for-bit comparable.
  const char* setup =
      "x = np.linspace(0.0, 4.5, 10)\n"
      "y = 2.0 - 1.5*x + 0.75*x*x + 0.1*np.sin(3.0*x)\n"
      "A = np.vander(x, 3, increasing=True)\n"  // columns [1, x, x^2]
      "sol, res, rank, sv = np.linalg.lstsq(A, y, rcond=-1)\n";

  ex::check("coefficients [c0,c1,c2]", c,
            std::string(setup) + "a = sol");
  ex::check("singular values of A", sv,
            std::string(setup) + "a = sv");

  // Effective rank: full column rank (3) since the columns are independent.
  ex::check_scalar("effective rank",
                   static_cast<double>(fit.rank.item<std::int64_t>({})), 3.0, 0.0);

  // Residual norm matches sqrt(sum of squared residuals) from NumPy.
  ndarray rn = ex::numpy(std::string(setup) + "a = np.array(np.linalg.norm(A@sol - y))");
  ex::check_scalar("residual 2-norm", resid_norm, rn.item<double>({}));

  // Sanity: the fit should be quite good (perturbation amplitude is 0.1).
  ex::check_scalar("residual is small", resid_norm < 0.3 ? 1.0 : 0.0, 1.0, 0.0);

  // Cross-check: reconstructing y from A and c yields the prediction; verify the
  // predicted values agree with NumPy's A @ sol elementwise.
  ndarray yhat = dot(A, c);
  ex::check("predicted A@c", yhat, std::string(setup) + "a = A @ sol");

  return ex::summary();
}
