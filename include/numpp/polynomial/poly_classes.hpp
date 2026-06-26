#pragma once

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"
#include "numpp/polynomial/orthopoly_calc.hpp"
#include "numpp/polynomial/orthopoly.hpp"
#include "numpp/polynomial/polynomial.hpp"

namespace numpp {
namespace polynomial {

// Orthogonal-polynomial classes (default domain/window = identity); evaluation,
// deriv, integ and roots delegate to the matching basis free functions.

class NUMPP_API Chebyshev {
 public:
  explicit Chebyshev(const ndarray& coef) : coef_(coef.astype(kFloat64).ravel().copy()) {}
  ndarray coef() const { return coef_; }
  ndarray operator()(const ndarray& x) const { return chebval(x, coef_); }
  Chebyshev deriv(int64_t m = 1) const { return Chebyshev(chebder(coef_, m)); }
  Chebyshev integ(int64_t m = 1, double k = 0.0) const { return Chebyshev(chebint(coef_, m, k)); }
  ndarray roots() const { return chebroots(coef_); }

 private:
  ndarray coef_;
};

class NUMPP_API Legendre {
 public:
  explicit Legendre(const ndarray& coef) : coef_(coef.astype(kFloat64).ravel().copy()) {}
  ndarray coef() const { return coef_; }
  ndarray operator()(const ndarray& x) const { return legval(x, coef_); }
  Legendre deriv(int64_t m = 1) const { return Legendre(legder(coef_, m)); }
  Legendre integ(int64_t m = 1, double k = 0.0) const { return Legendre(legint(coef_, m, k)); }
  ndarray roots() const { return legroots(coef_); }

 private:
  ndarray coef_;
};

class NUMPP_API Hermite {
 public:
  explicit Hermite(const ndarray& coef) : coef_(coef.astype(kFloat64).ravel().copy()) {}
  ndarray coef() const { return coef_; }
  ndarray operator()(const ndarray& x) const { return hermval(x, coef_); }
  Hermite deriv(int64_t m = 1) const { return Hermite(hermder(coef_, m)); }
  Hermite integ(int64_t m = 1, double k = 0.0) const { return Hermite(hermint(coef_, m, k)); }
  ndarray roots() const { return hermroots(coef_); }

 private:
  ndarray coef_;
};

class NUMPP_API HermiteE {
 public:
  explicit HermiteE(const ndarray& coef) : coef_(coef.astype(kFloat64).ravel().copy()) {}
  ndarray coef() const { return coef_; }
  ndarray operator()(const ndarray& x) const { return hermeval(x, coef_); }
  HermiteE deriv(int64_t m = 1) const { return HermiteE(hermeder(coef_, m)); }
  HermiteE integ(int64_t m = 1, double k = 0.0) const { return HermiteE(hermeint(coef_, m, k)); }
  ndarray roots() const { return hermeroots(coef_); }

 private:
  ndarray coef_;
};

class NUMPP_API Laguerre {
 public:
  explicit Laguerre(const ndarray& coef) : coef_(coef.astype(kFloat64).ravel().copy()) {}
  ndarray coef() const { return coef_; }
  ndarray operator()(const ndarray& x) const { return lagval(x, coef_); }
  Laguerre deriv(int64_t m = 1) const { return Laguerre(lagder(coef_, m)); }
  Laguerre integ(int64_t m = 1, double k = 0.0) const { return Laguerre(lagint(coef_, m, k)); }
  ndarray roots() const { return lagroots(coef_); }

 private:
  ndarray coef_;
};

}  // namespace polynomial
}  // namespace numpp
