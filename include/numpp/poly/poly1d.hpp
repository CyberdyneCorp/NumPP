#pragma once

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// numpy.poly1d: a 1-D polynomial with coefficients in HIGHEST-degree-first order.
class NUMPP_API poly1d {
 public:
  explicit poly1d(const ndarray& coeffs);  // highest-first; leading zeros trimmed

  ndarray coeffs() const;
  int64_t order() const;                    // degree
  ndarray operator()(const ndarray& x) const;  // evaluate
  poly1d deriv(int64_t m = 1) const;
  poly1d integ(int64_t m = 1, double k = 0.0) const;
  ndarray roots() const;

  poly1d operator+(const poly1d& o) const;
  poly1d operator-(const poly1d& o) const;
  poly1d operator*(const poly1d& o) const;

 private:
  ndarray coeffs_;  // highest-first
};

// Weighted least-squares polynomial fit (numpy.polyfit with w=), highest-first.
NUMPP_API ndarray polyfit_weighted(const ndarray& x, const ndarray& y, int64_t deg, const ndarray& w);

}  // namespace numpp
