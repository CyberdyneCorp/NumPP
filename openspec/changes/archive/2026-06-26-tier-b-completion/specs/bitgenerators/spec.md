# bitgenerators Specification

## ADDED Requirements

### Requirement: Philox and SFC64 BitGenerators
NumPP SHALL provide Philox and SFC64 BitGenerators seeded from SeedSequence,
producing a deterministic uint64 stream.

#### Scenario: SFC64 stream is bit-exact with numpy
- WHEN SFC64(seed).random_raw(n) is generated
- THEN the uint64 values equal numpy.random.SFC64(seed).random_raw(n)

#### Scenario: Philox is deterministic and seed-sensitive
- WHEN Philox(seed).random_raw is generated twice with the same seed
- THEN the streams are identical, and differ for a different seed
- AND bit-exact parity with numpy's Philox stream is tracked separately (#36)
