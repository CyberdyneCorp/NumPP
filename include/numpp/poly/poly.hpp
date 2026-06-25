#pragma once

#include <string>
#include <utility>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- signal ----
// mode: "full" | "same" | "valid"
NUMPP_API ndarray convolve(const ndarray& a, const ndarray& v, const std::string& mode = "full");
NUMPP_API ndarray correlate(const ndarray& a, const ndarray& v, const std::string& mode = "valid");
NUMPP_API ndarray interp(const ndarray& x, const ndarray& xp, const ndarray& fp);  // 1-D linear

// ---- polynomials (coefficients highest power first, like numpy.poly*) ----
NUMPP_API ndarray polyval(const ndarray& p, const ndarray& x);
NUMPP_API ndarray polyadd(const ndarray& a, const ndarray& b);
NUMPP_API ndarray polysub(const ndarray& a, const ndarray& b);
NUMPP_API ndarray polymul(const ndarray& a, const ndarray& b);
NUMPP_API std::pair<ndarray, ndarray> polydiv(const ndarray& a, const ndarray& b);  // (quotient, remainder)
NUMPP_API ndarray polyder(const ndarray& p, int64_t m = 1);
NUMPP_API ndarray polyint(const ndarray& p, int64_t m = 1, double k = 0.0);
NUMPP_API ndarray poly(const ndarray& roots);     // roots -> coefficients
NUMPP_API ndarray roots(const ndarray& p);        // coefficients -> roots (complex128)
NUMPP_API ndarray polyfit(const ndarray& x, const ndarray& y, int64_t deg);

}  // namespace numpp
