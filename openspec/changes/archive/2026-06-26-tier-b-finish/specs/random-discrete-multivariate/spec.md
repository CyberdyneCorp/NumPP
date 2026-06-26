# random-discrete-multivariate Specification

## MODIFIED Requirements

### Requirement: Discrete, continuous and multivariate distributions
NumPP SHALL provide geometric, negative_binomial, zipf, logseries, standard_t, f,
wald, vonmises, noncentral_chisquare, multinomial, multivariate_normal, dirichlet,
AND hypergeometric, noncentral_f and bytes over a Generator, using numpy's
parameterisation.

#### Scenario: distributions match theoretical moments
- WHEN a large sample is drawn from a distribution
- THEN sample moments and support constraints match the closed-form theory within
  statistical tolerance

### Requirement: Non-bit-exact disclosure
These distributions SHALL be statistically correct but are not guaranteed
bit-identical to numpy's sample values (issue #8).

#### Scenario: documented as statistically correct
- WHEN a caller relies on these distributions
- THEN they receive statistically correct samples, documented as not bit-exact
