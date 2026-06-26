# Add random distributions (NumPy parity, Tier 2)

## Why
Ninth item of the numpy-parity-roadmap backlog. NumPy exposes many continuous
distributions beyond the base set; the common inverse-CDF family was missing.

## What changes
- **random-distributions** capability (free functions over a `Generator&`):
  laplace, logistic, gumbel, rayleigh, weibull, pareto (Lomax), power,
  standard_cauchy, triangular, lognormal. Drawn from the Generator's bit-exact
  uniform stream via inverse-CDF / standard transforms.

## Non-goals
- **Bit-exact** equality with numpy's sample values: like the base Generator's
  gaussian path (issue #8), these are statistically correct but not bit-identical
  to numpy. Validated by large-sample moment tests against closed-form theory.
- Philox / SFC64 BitGenerators, and the discrete / multivariate distributions
  (hypergeometric, negative_binomial, multinomial, multivariate_normal,
  dirichlet, vonmises, wald, standard_t, f, noncentral_*) — deferred.
