#pragma once

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"
#include "numpp/random/random.hpp"

namespace numpp {
namespace random {

// Additional continuous distributions, drawn from the Generator's (bit-exact)
// uniform stream via inverse-CDF / standard transforms. Statistically correct
// and matching numpy's parameterisation, but — like the base Generator's
// gaussian path — not bit-identical to numpy's sample values (see issue #8).

NUMPP_API ndarray laplace(Generator& g, double loc, double scale, const Shape& size);
NUMPP_API ndarray logistic(Generator& g, double loc, double scale, const Shape& size);
NUMPP_API ndarray gumbel(Generator& g, double loc, double scale, const Shape& size);
NUMPP_API ndarray rayleigh(Generator& g, double scale, const Shape& size);
NUMPP_API ndarray weibull(Generator& g, double a, const Shape& size);
NUMPP_API ndarray pareto(Generator& g, double a, const Shape& size);       // Lomax (numpy convention)
NUMPP_API ndarray power(Generator& g, double a, const Shape& size);
NUMPP_API ndarray standard_cauchy(Generator& g, const Shape& size);
NUMPP_API ndarray triangular(Generator& g, double left, double mode, double right, const Shape& size);
NUMPP_API ndarray lognormal(Generator& g, double mean, double sigma, const Shape& size);

}  // namespace random
}  // namespace numpp
