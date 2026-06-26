#pragma once

#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace polynomial {

// numpy.polynomial.* uses coefficients in LOWEST-degree-first order (the opposite
// of the legacy numpy.poly* API). All functions here follow that convention.

// ---- power-basis free functions (numpy.polynomial.polynomial.*) ----
NUMPP_API ndarray polyval(const ndarray& x, const ndarray& c);
NUMPP_API ndarray polyadd(const ndarray& a, const ndarray& b);
NUMPP_API ndarray polysub(const ndarray& a, const ndarray& b);
NUMPP_API ndarray polymul(const ndarray& a, const ndarray& b);
NUMPP_API ndarray polyder(const ndarray& c, int64_t m = 1);
NUMPP_API ndarray polyint(const ndarray& c, int64_t m = 1, double k = 0.0);
NUMPP_API ndarray polyroots(const ndarray& c);
NUMPP_API ndarray polyfit(const ndarray& x, const ndarray& y, int64_t deg);

// ---- orthogonal basis evaluation (numpy.polynomial.<basis>.<basis>val) ----
NUMPP_API ndarray chebval(const ndarray& x, const ndarray& c);   // Chebyshev T
NUMPP_API ndarray legval(const ndarray& x, const ndarray& c);    // Legendre
NUMPP_API ndarray hermval(const ndarray& x, const ndarray& c);   // physicists' Hermite
NUMPP_API ndarray hermeval(const ndarray& x, const ndarray& c);  // probabilists' Hermite
NUMPP_API ndarray lagval(const ndarray& x, const ndarray& c);    // Laguerre

// ---- power-basis Polynomial class (subset of numpy.polynomial.Polynomial) ----
class NUMPP_API Polynomial {
 public:
  explicit Polynomial(const ndarray& coef);
  explicit Polynomial(std::vector<double> coef);

  ndarray coef() const;
  ndarray operator()(const ndarray& x) const;
  Polynomial deriv(int64_t m = 1) const;
  Polynomial integ(int64_t m = 1, double k = 0.0) const;
  ndarray roots() const;
  Polynomial operator+(const Polynomial& o) const;
  Polynomial operator-(const Polynomial& o) const;
  Polynomial operator*(const Polynomial& o) const;

  static Polynomial fit(const ndarray& x, const ndarray& y, int64_t deg);

 private:
  std::vector<double> coef_;  // lowest-first
};

}  // namespace polynomial
}  // namespace numpp
