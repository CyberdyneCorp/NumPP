# random-distributions Specification

## Purpose
TBD - created by archiving change add-random-distributions. Update Purpose after archive.
## Requirements
### Requirement: Continuous distribution sampling
NumPP SHALL provide laplace, logistic, gumbel, rayleigh, weibull, pareto (Lomax),
power, standard_cauchy, triangular and lognormal sampling functions over a
Generator, using numpy's parameterisation.

#### Scenario: sampled distributions match theoretical moments
- WHEN a large sample is drawn from a distribution with given parameters
- THEN the sample mean and standard deviation match the closed-form theoretical
  moments within statistical tolerance

#### Scenario: bounded supports are respected
- WHEN power or triangular is sampled
- THEN all samples fall within the distribution's support

### Requirement: Non-bit-exact disclosure
The distribution functions SHALL be statistically correct but are not guaranteed
bit-identical to numpy's sample values (consistent with the Generator's gaussian
path, issue #8).

#### Scenario: documented as statistically correct, not bit-exact
- WHEN a caller relies on these distribution functions
- THEN they receive statistically correct samples
- AND the API documents that values are not bit-identical to numpy

