#pragma once

#include <utility>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- products (np.* namespace) ----
NUMPP_API ndarray dot(const ndarray& a, const ndarray& b);
NUMPP_API ndarray vdot(const ndarray& a, const ndarray& b);
NUMPP_API ndarray inner(const ndarray& a, const ndarray& b);
NUMPP_API ndarray outer(const ndarray& a, const ndarray& b);
NUMPP_API ndarray trace(const ndarray& a, int64_t offset = 0);
NUMPP_API ndarray kron(const ndarray& a, const ndarray& b);

namespace linalg {

struct SignLogDet { ndarray sign; ndarray logabsdet; };

NUMPP_API ndarray solve(const ndarray& a, const ndarray& b);
NUMPP_API ndarray inv(const ndarray& a);
NUMPP_API ndarray det(const ndarray& a);
NUMPP_API SignLogDet slogdet(const ndarray& a);
NUMPP_API ndarray matrix_power(const ndarray& a, int64_t n);
NUMPP_API ndarray cholesky(const ndarray& a);

}  // namespace linalg
}  // namespace numpp
