#pragma once

#include <array>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"
#include "numpp/polynomial/orthopoly_calc.hpp"
#include "numpp/polynomial/orthopoly.hpp"
#include "numpp/polynomial/polynomial.hpp"

namespace numpp {
namespace polynomial {

namespace detail {
// Affine-map x from `domain` into `window` (numpy.polynomial mapdomain).
NUMPP_API ndarray map_domain(const ndarray& x, const std::array<double, 2>& domain,
                             const std::array<double, 2>& window);
// Least-squares solve of the basis Vandermonde system V c = y; returns 1-D c.
NUMPP_API ndarray ortho_lstsq(const ndarray& V, const ndarray& y);
// [min, max] of x — the default fit domain.
NUMPP_API std::array<double, 2> data_range(const ndarray& x);
}  // namespace detail

// Orthogonal-polynomial classes with numpy domain/window mapping. Inputs are
// affine-mapped from `domain` into `window` before evaluation; the defaults make
// the map the identity, so a plain construction matches the bare basis. `fit`
// least-squares fits in the mapped coordinates, defaulting `domain` to the data
// range (matching numpy.polynomial.*.fit).

class NUMPP_API Chebyshev {
 public:
  explicit Chebyshev(const ndarray& coef, std::array<double, 2> domain = {-1, 1},
                     std::array<double, 2> window = {-1, 1})
      : coef_(coef.astype(kFloat64).ravel().copy()), domain_(domain), window_(window) {}
  ndarray coef() const { return coef_; }
  std::array<double, 2> domain() const { return domain_; }
  std::array<double, 2> window() const { return window_; }
  ndarray operator()(const ndarray& x) const { return chebval(detail::map_domain(x, domain_, window_), coef_); }
  Chebyshev deriv(int64_t m = 1) const { return Chebyshev(chebder(coef_, m), domain_, window_); }
  Chebyshev integ(int64_t m = 1, double k = 0.0) const { return Chebyshev(chebint(coef_, m, k), domain_, window_); }
  ndarray roots() const { return chebroots(coef_); }
  static Chebyshev fit(const ndarray& x, const ndarray& y, int64_t deg) {
    auto dom = detail::data_range(x);
    std::array<double, 2> win{-1, 1};
    ndarray c = detail::ortho_lstsq(chebvander(detail::map_domain(x, dom, win), deg), y);
    return Chebyshev(c, dom, win);
  }

 private:
  ndarray coef_;
  std::array<double, 2> domain_, window_;
};

class NUMPP_API Legendre {
 public:
  explicit Legendre(const ndarray& coef, std::array<double, 2> domain = {-1, 1},
                    std::array<double, 2> window = {-1, 1})
      : coef_(coef.astype(kFloat64).ravel().copy()), domain_(domain), window_(window) {}
  ndarray coef() const { return coef_; }
  std::array<double, 2> domain() const { return domain_; }
  std::array<double, 2> window() const { return window_; }
  ndarray operator()(const ndarray& x) const { return legval(detail::map_domain(x, domain_, window_), coef_); }
  Legendre deriv(int64_t m = 1) const { return Legendre(legder(coef_, m), domain_, window_); }
  Legendre integ(int64_t m = 1, double k = 0.0) const { return Legendre(legint(coef_, m, k), domain_, window_); }
  ndarray roots() const { return legroots(coef_); }
  static Legendre fit(const ndarray& x, const ndarray& y, int64_t deg) {
    auto dom = detail::data_range(x);
    std::array<double, 2> win{-1, 1};
    ndarray c = detail::ortho_lstsq(legvander(detail::map_domain(x, dom, win), deg), y);
    return Legendre(c, dom, win);
  }

 private:
  ndarray coef_;
  std::array<double, 2> domain_, window_;
};

class NUMPP_API Hermite {
 public:
  explicit Hermite(const ndarray& coef, std::array<double, 2> domain = {-1, 1},
                   std::array<double, 2> window = {-1, 1})
      : coef_(coef.astype(kFloat64).ravel().copy()), domain_(domain), window_(window) {}
  ndarray coef() const { return coef_; }
  std::array<double, 2> domain() const { return domain_; }
  std::array<double, 2> window() const { return window_; }
  ndarray operator()(const ndarray& x) const { return hermval(detail::map_domain(x, domain_, window_), coef_); }
  Hermite deriv(int64_t m = 1) const { return Hermite(hermder(coef_, m), domain_, window_); }
  Hermite integ(int64_t m = 1, double k = 0.0) const { return Hermite(hermint(coef_, m, k), domain_, window_); }
  ndarray roots() const { return hermroots(coef_); }
  static Hermite fit(const ndarray& x, const ndarray& y, int64_t deg) {
    auto dom = detail::data_range(x);
    std::array<double, 2> win{-1, 1};
    ndarray c = detail::ortho_lstsq(hermvander(detail::map_domain(x, dom, win), deg), y);
    return Hermite(c, dom, win);
  }

 private:
  ndarray coef_;
  std::array<double, 2> domain_, window_;
};

class NUMPP_API HermiteE {
 public:
  explicit HermiteE(const ndarray& coef, std::array<double, 2> domain = {-1, 1},
                    std::array<double, 2> window = {-1, 1})
      : coef_(coef.astype(kFloat64).ravel().copy()), domain_(domain), window_(window) {}
  ndarray coef() const { return coef_; }
  std::array<double, 2> domain() const { return domain_; }
  std::array<double, 2> window() const { return window_; }
  ndarray operator()(const ndarray& x) const { return hermeval(detail::map_domain(x, domain_, window_), coef_); }
  HermiteE deriv(int64_t m = 1) const { return HermiteE(hermeder(coef_, m), domain_, window_); }
  HermiteE integ(int64_t m = 1, double k = 0.0) const { return HermiteE(hermeint(coef_, m, k), domain_, window_); }
  ndarray roots() const { return hermeroots(coef_); }
  static HermiteE fit(const ndarray& x, const ndarray& y, int64_t deg) {
    auto dom = detail::data_range(x);
    std::array<double, 2> win{-1, 1};
    ndarray c = detail::ortho_lstsq(hermevander(detail::map_domain(x, dom, win), deg), y);
    return HermiteE(c, dom, win);
  }

 private:
  ndarray coef_;
  std::array<double, 2> domain_, window_;
};

class NUMPP_API Laguerre {
 public:
  explicit Laguerre(const ndarray& coef, std::array<double, 2> domain = {0, 1},
                    std::array<double, 2> window = {0, 1})
      : coef_(coef.astype(kFloat64).ravel().copy()), domain_(domain), window_(window) {}
  ndarray coef() const { return coef_; }
  std::array<double, 2> domain() const { return domain_; }
  std::array<double, 2> window() const { return window_; }
  ndarray operator()(const ndarray& x) const { return lagval(detail::map_domain(x, domain_, window_), coef_); }
  Laguerre deriv(int64_t m = 1) const { return Laguerre(lagder(coef_, m), domain_, window_); }
  Laguerre integ(int64_t m = 1, double k = 0.0) const { return Laguerre(lagint(coef_, m, k), domain_, window_); }
  ndarray roots() const { return lagroots(coef_); }
  static Laguerre fit(const ndarray& x, const ndarray& y, int64_t deg) {
    auto dom = detail::data_range(x);
    std::array<double, 2> win{0, 1};
    ndarray c = detail::ortho_lstsq(lagvander(detail::map_domain(x, dom, win), deg), y);
    return Laguerre(c, dom, win);
  }

 private:
  ndarray coef_;
  std::array<double, 2> domain_, window_;
};

}  // namespace polynomial
}  // namespace numpp
