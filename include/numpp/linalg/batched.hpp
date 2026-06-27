#pragma once

// Internal: batched (stacked) linear algebra. Each entry treats the last two
// axes of an N-D array as a matrix and applies the matching 2-D routine to every
// matrix in the leading-axis stack, restacking the results (numpy.linalg
// semantics). Only used by linalg.cpp when ndim() > 2.

#include <string>

#include "numpp/core/ndarray.hpp"
#include "numpp/linalg/linalg.hpp"

namespace numpp {
namespace linalg {
namespace batched {

ndarray solve(const ndarray& a, const ndarray& b);
ndarray inv(const ndarray& a);
ndarray det(const ndarray& a);
SignLogDet slogdet(const ndarray& a);
ndarray matrix_power(const ndarray& a, int64_t n);
ndarray cholesky(const ndarray& a);
QRResult qr(const ndarray& a, const std::string& mode);
ndarray eigvalsh(const ndarray& a);
EighResult eigh(const ndarray& a);
ndarray eigvals(const ndarray& a);
EigResult eig(const ndarray& a);
SVDResult svd(const ndarray& a, bool full_matrices);
ndarray svdvals(const ndarray& a);
ndarray pinv(const ndarray& a, double rcond);
ndarray matrix_rank(const ndarray& a);

}  // namespace batched
}  // namespace linalg
}  // namespace numpp
