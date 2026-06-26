# random Specification

## ADDED Requirements

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
