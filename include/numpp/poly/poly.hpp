#pragma once

#include <optional>
#include <string>
#include <utility>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- signal ----
// mode: "full" | "same" | "valid"
NUMPP_API ndarray convolve(const ndarray& a, const ndarray& v, const std::string& mode = "full");
NUMPP_API ndarray correlate(const ndarray& a, const ndarray& v, const std::string& mode = "valid");
// 1-D linear interpolation (numpy.interp). `left`/`right` are the fill values for
// x below xp[0] / above xp[-1] (default fp[0]/fp[-1]); `period` enables periodic
// interpolation (xp need not be sorted; left/right are then ignored).
NUMPP_API ndarray interp(const ndarray& x, const ndarray& xp, const ndarray& fp,
                         std::optional<double> left = std::nullopt,
                         std::optional<double> right = std::nullopt,
                         std::optional<double> period = std::nullopt);

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
