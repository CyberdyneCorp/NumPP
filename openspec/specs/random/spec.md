# random Specification

## Purpose
TBD - created by archiving change add-random. Update Purpose after archive.
## Requirements
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

### Requirement: Bit-exact Philox and MT19937 BitGenerators
NumPP SHALL provide Philox and MT19937 BitGenerators whose raw streams are
bit-identical to numpy.random.Philox(seed).random_raw() and
numpy.random.MT19937(seed).random_raw(), seeded from the (bit-exact) SeedSequence.

#### Scenario: Philox stream matches numpy
- WHEN Philox(seed).random_raw(n) is generated
- THEN every uint64 equals numpy.random.Philox(seed).random_raw(n)

#### Scenario: MT19937 stream matches numpy
- WHEN MT19937BitGen(seed).random_raw(n) is generated
- THEN every value equals numpy.random.MT19937(seed).random_raw(n)

### Requirement: Bit-exact ziggurat normals/exponentials and choice without replacement
NumPP's Generator SHALL produce standard_normal, standard_exponential, normal and
choice(replace=False) bit-identical to numpy.random.default_rng(seed) for the same
seed, using numpy's ziggurat (with its lookup tables) and Floyd/tail-shuffle
selection algorithms.

#### Scenario: ziggurat samples match numpy
- WHEN standard_normal / standard_exponential / normal are drawn from a seeded
  Generator
- THEN every value equals numpy.random.default_rng(seed)'s corresponding draw

#### Scenario: choice without replacement matches numpy
- WHEN choice(pop, size, replace=False) is drawn (both the Floyd and the
  large-population tail-shuffle paths)
- THEN the selected indices equal numpy.random.default_rng(seed).choice(...)

