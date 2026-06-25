# random Specification

## ADDED Requirements

### Requirement: Bit-exact BitGenerators

NumPP SHALL provide `SeedSequence`, `PCG64`, and `MT19937` reproducing numpy's
algorithms exactly: for a given integer seed, `SeedSequence.generate_state`, the
PCG64 raw 64-bit stream, and the MT19937 raw 32-bit stream SHALL equal numpy's.

#### Scenario: PCG64 raw stream matches numpy
- WHEN a `Generator(seed)` produces `random_raw(k)`
- THEN the values equal `numpy.random.PCG64(seed).random_raw(k)` exactly

#### Scenario: SeedSequence state matches numpy
- WHEN `SeedSequence(seed).generate_state(n)` is computed
- THEN it equals `numpy.random.SeedSequence(seed).generate_state(n)` exactly

### Requirement: Generator core sampling

NumPP SHALL provide a `Generator` with `random` (uniform doubles in [0,1) via
numpy's 53-bit method), `integers` (Lemire bounded), `uniform`, and the
array-shaping/permutation operations `shuffle`, `permutation`, `choice`, matching
`numpy.random.Generator(PCG64(seed))` exactly for the reproducible methods.

#### Scenario: random doubles match numpy
- WHEN `Generator(seed).random(shape)` is computed
- THEN it equals `numpy.random.Generator(np.random.PCG64(seed)).random(shape)` exactly

#### Scenario: integers match numpy
- WHEN `Generator(seed).integers(low, high, size)` is computed
- THEN it equals `numpy.random.Generator(np.random.PCG64(seed)).integers(low, high, size)` exactly

### Requirement: Distributions

NumPP SHALL provide the common distributions (`standard_normal`, `normal`,
`exponential`, `standard_exponential`, `poisson`, `binomial`, `gamma`, `beta`,
`chisquare`). Where numpy's sampler is reproducible in portable C++ the result
SHALL be bit-exact; otherwise it SHALL pass distributional (KS / moment) checks,
with the non-bit-exactness recorded as an issue.

#### Scenario: Normal samples are correctly distributed
- WHEN a large `standard_normal` sample is drawn
- THEN its mean is ~0 and variance ~1 within statistical tolerance (and bit-exact
  vs numpy where the ziggurat tables are reproduced)

### Requirement: Legacy RandomState

NumPP SHALL provide a legacy `RandomState` (MT19937-backed) with `rand`, `randn`,
`randint`, `random_sample`, and `normal`, matching `numpy.random.RandomState(seed)`.

#### Scenario: RandomState reproducibility
- WHEN `RandomState(seed).random_sample(k)` is computed
- THEN it equals `numpy.random.RandomState(seed).random_sample(k)` exactly
