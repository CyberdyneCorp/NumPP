# ufunc-completion Specification

## ADDED Requirements

### Requirement: Elementwise closeness and infinity predicates
NumPP SHALL provide `isclose(a, b, rtol, atol, equal_nan)` returning a boolean
array (the elementwise form of `allclose`), and `isposinf` / `isneginf` returning
boolean arrays, all matching numpy including broadcasting and NaN handling.

#### Scenario: predicates match numpy elementwise
- WHEN `isclose` compares two broadcastable arrays (with and without `equal_nan`)
- THEN the boolean array equals `numpy.isclose` with the same tolerances
- WHEN `isposinf` / `isneginf` are applied to an array containing `+inf`/`-inf`/NaN/finite
- THEN each boolean array equals `numpy.isposinf` / `numpy.isneginf`
