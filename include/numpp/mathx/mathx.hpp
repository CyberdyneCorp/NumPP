#pragma once

#include <optional>
#include <utility>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- rounding / angle ----
NUMPP_API ndarray around(const ndarray& a, int64_t decimals = 0);  // round half to even
NUMPP_API ndarray degrees(const ndarray& a);                       // alias of rad2deg
NUMPP_API ndarray radians(const ndarray& a);                       // alias of deg2rad
NUMPP_API ndarray sinc(const ndarray& a);                          // sin(pi x)/(pi x)

// ---- integer ----
NUMPP_API ndarray gcd(const ndarray& a, const ndarray& b);
NUMPP_API ndarray lcm(const ndarray& a, const ndarray& b);

// ---- nan / inf handling ----
NUMPP_API ndarray nan_to_num(const ndarray& a, double nan = 0.0,
                             std::optional<double> posinf = std::nullopt,
                             std::optional<double> neginf = std::nullopt);

// ---- log-sum-exp / powers / mod ----
NUMPP_API ndarray logaddexp(const ndarray& a, const ndarray& b);
NUMPP_API ndarray logaddexp2(const ndarray& a, const ndarray& b);
NUMPP_API ndarray float_power(const ndarray& a, const ndarray& b);
NUMPP_API ndarray fmod(const ndarray& a, const ndarray& b);
NUMPP_API ndarray heaviside(const ndarray& a, const ndarray& h0);

// ---- mantissa / exponent decompositions ----
NUMPP_API std::pair<ndarray, ndarray> modf(const ndarray& a);    // (fractional, integral)
NUMPP_API std::pair<ndarray, ndarray> frexp(const ndarray& a);   // (mantissa, exponent int64)
NUMPP_API ndarray ldexp(const ndarray& m, const ndarray& e);
NUMPP_API std::pair<ndarray, ndarray> divmod(const ndarray& a, const ndarray& b);  // (floor div, mod)

// ---- misc ----
NUMPP_API ndarray unwrap(const ndarray& p);          // 1-D phase unwrap, period 2*pi
NUMPP_API ndarray i0(const ndarray& a);              // modified Bessel I0
NUMPP_API ndarray nextafter(const ndarray& a, const ndarray& b);
NUMPP_API ndarray spacing(const ndarray& a);

}  // namespace numpp
