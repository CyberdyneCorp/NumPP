#pragma once

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace polynomial {

// Derivative and antiderivative of an orthogonal-basis series (lowest-first),
// matching numpy.polynomial.<basis>.<basis>der / <basis>int (default scl=1,
// integration lower bound 0).

NUMPP_API ndarray chebder(const ndarray& c, int64_t m = 1);
NUMPP_API ndarray chebint(const ndarray& c, int64_t m = 1, double k = 0.0);
NUMPP_API ndarray legder(const ndarray& c, int64_t m = 1);
NUMPP_API ndarray legint(const ndarray& c, int64_t m = 1, double k = 0.0);
NUMPP_API ndarray hermder(const ndarray& c, int64_t m = 1);   // physicists'
NUMPP_API ndarray hermint(const ndarray& c, int64_t m = 1, double k = 0.0);
NUMPP_API ndarray hermeder(const ndarray& c, int64_t m = 1);  // probabilists'
NUMPP_API ndarray hermeint(const ndarray& c, int64_t m = 1, double k = 0.0);
NUMPP_API ndarray lagder(const ndarray& c, int64_t m = 1);
NUMPP_API ndarray lagint(const ndarray& c, int64_t m = 1, double k = 0.0);

}  // namespace polynomial
}  // namespace numpp
