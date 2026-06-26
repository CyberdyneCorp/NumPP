# random Specification

## ADDED Requirements

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
