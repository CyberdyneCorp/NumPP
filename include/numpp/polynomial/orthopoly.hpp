#pragma once

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace polynomial {

// Vandermonde matrices and roots for each orthogonal basis, matching
// numpy.polynomial.<basis>.<basis>vander / <basis>roots. Coefficients and roots
// follow numpy's lowest-degree-first convention. vander(x, deg) returns
// (len(x), deg+1) with column j = B_j(x). roots(c) returns the roots of the
// series sum_j c[j] B_j via the basis companion matrix.

NUMPP_API ndarray chebvander(const ndarray& x, int64_t deg);
NUMPP_API ndarray chebroots(const ndarray& c);
NUMPP_API ndarray legvander(const ndarray& x, int64_t deg);
NUMPP_API ndarray legroots(const ndarray& c);
NUMPP_API ndarray hermvander(const ndarray& x, int64_t deg);   // physicists'
NUMPP_API ndarray hermroots(const ndarray& c);
NUMPP_API ndarray hermevander(const ndarray& x, int64_t deg);  // probabilists'
NUMPP_API ndarray hermeroots(const ndarray& c);
NUMPP_API ndarray lagvander(const ndarray& x, int64_t deg);
NUMPP_API ndarray lagroots(const ndarray& c);

}  // namespace polynomial
}  // namespace numpp
