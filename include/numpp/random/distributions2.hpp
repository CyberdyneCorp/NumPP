#pragma once

#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"
#include "numpp/random/random.hpp"

namespace numpp {
namespace random {

// Discrete, continuous and multivariate distributions built on the Generator's
// primitives (uniform/normal/standard_exponential/gamma/chisquare/poisson).
// Statistically correct, matching numpy's parameterisation; not bit-exact (#8).

// ---- discrete ----
NUMPP_API ndarray geometric(Generator& g, double p, const Shape& size);
NUMPP_API ndarray negative_binomial(Generator& g, double n, double p, const Shape& size);
NUMPP_API ndarray zipf(Generator& g, double a, const Shape& size);
NUMPP_API ndarray logseries(Generator& g, double p, const Shape& size);

// ---- continuous ----
NUMPP_API ndarray standard_t(Generator& g, double df, const Shape& size);
NUMPP_API ndarray f(Generator& g, double dfnum, double dfden, const Shape& size);
NUMPP_API ndarray wald(Generator& g, double mean, double scale, const Shape& size);
NUMPP_API ndarray vonmises(Generator& g, double mu, double kappa, const Shape& size);
NUMPP_API ndarray noncentral_chisquare(Generator& g, double df, double nonc, const Shape& size);

// ---- multivariate (return shape noted) ----
NUMPP_API ndarray multinomial(Generator& g, int64_t n, const std::vector<double>& pvals);          // (k,) counts
NUMPP_API ndarray multivariate_normal(Generator& g, const ndarray& mean, const ndarray& cov, int64_t size);  // (size, d)
NUMPP_API ndarray dirichlet(Generator& g, const std::vector<double>& alpha, int64_t size);          // (size, k)

}  // namespace random
}  // namespace numpp
