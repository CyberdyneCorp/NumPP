# ufunc-completion Specification

## Purpose
TBD - created by archiving change tier-a-completion. Update Purpose after archive.
## Requirements
### Requirement: Rounding, type inspection and bit packing
NumPP SHALL provide fix, real_if_close, iscomplexobj, isrealobj, iscomplex,
isreal, common_type, packbits and unpackbits matching numpy.

#### Scenario: ufunc/type helpers match numpy
- WHEN fix/real_if_close/iscomplex/isreal/packbits/unpackbits are evaluated
- THEN the result equals the corresponding numpy function
- AND iscomplexobj/isrealobj/common_type report numpy's type classification

### Requirement: Complex-promoting math (emath)
NumPP SHALL provide emath::sqrt, emath::log and emath::power that promote to
complex for out-of-domain inputs, matching numpy.emath.

#### Scenario: emath matches numpy
- WHEN emath::sqrt/log/power are evaluated on inputs outside the real domain
- THEN the result equals numpy.emath.sqrt / numpy.emath.log / numpy.emath.power

### Requirement: Elementwise closeness and infinity predicates
NumPP SHALL provide `isclose(a, b, rtol, atol, equal_nan)` returning a boolean
array (the elementwise form of `allclose`), and `isposinf` / `isneginf` returning
boolean arrays, all matching numpy including broadcasting and NaN handling.

#### Scenario: predicates match numpy elementwise
- WHEN `isclose` compares two broadcastable arrays (with and without `equal_nan`)
- THEN the boolean array equals `numpy.isclose` with the same tolerances
- WHEN `isposinf` / `isneginf` are applied to an array containing `+inf`/`-inf`/NaN/finite
- THEN each boolean array equals `numpy.isposinf` / `numpy.isneginf`

