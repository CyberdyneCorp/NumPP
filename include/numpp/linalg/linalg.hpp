#pragma once

#include <string>
#include <utility>
#include <vector>

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
struct QRResult { ndarray q; ndarray r; };
struct EighResult { ndarray eigenvalues; ndarray eigenvectors; };
struct SVDResult { ndarray u; ndarray s; ndarray vh; };
struct EigResult { ndarray eigenvalues; ndarray eigenvectors; };
struct LstsqResult { ndarray solution; ndarray residuals; ndarray rank; ndarray singular_values; };

NUMPP_API ndarray solve(const ndarray& a, const ndarray& b);
NUMPP_API ndarray inv(const ndarray& a);
NUMPP_API ndarray det(const ndarray& a);
NUMPP_API SignLogDet slogdet(const ndarray& a);
NUMPP_API ndarray matrix_power(const ndarray& a, int64_t n);
NUMPP_API ndarray cholesky(const ndarray& a);

// mode: "reduced" (default) or "complete".
NUMPP_API QRResult qr(const ndarray& a, const std::string& mode = "reduced");

// Hermitian/symmetric eigensolver, eigenvalues ascending.
NUMPP_API EighResult eigh(const ndarray& a);
NUMPP_API ndarray eigvalsh(const ndarray& a);

// General (non-symmetric) eigensolver. Eigenvalues/eigenvectors may be complex.
NUMPP_API ndarray eigvals(const ndarray& a);
NUMPP_API EigResult eig(const ndarray& a);

NUMPP_API SVDResult svd(const ndarray& a, bool full_matrices = true);
NUMPP_API ndarray svdvals(const ndarray& a);                 // singular values, descending
NUMPP_API ndarray pinv(const ndarray& a, double rcond = 1e-15);
NUMPP_API ndarray matrix_rank(const ndarray& a);
NUMPP_API LstsqResult lstsq(const ndarray& a, const ndarray& b, double rcond = -1.0);

// numpy.linalg.tensorsolve / tensorinv: solve a x = b / invert a tensor by
// reshaping the chosen axes into a square matrix.
NUMPP_API ndarray tensorsolve(const ndarray& a, const ndarray& b,
                              const std::vector<int64_t>& axes = {});
NUMPP_API ndarray tensorinv(const ndarray& a, int64_t ind = 2);

// Norms. Default: 2-norm (vector) / Frobenius (matrix). String ords: "fro","nuc".
NUMPP_API ndarray norm(const ndarray& a);
NUMPP_API ndarray norm(const ndarray& a, double ord);
NUMPP_API ndarray norm(const ndarray& a, const std::string& ord);

}  // namespace linalg
}  // namespace numpp
