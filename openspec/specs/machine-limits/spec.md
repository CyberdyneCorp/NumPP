# machine-limits Specification

## Purpose
TBD - created by archiving change numpy-foundation-completion. Update Purpose after archive.
## Requirements
### Requirement: Floating-point and integer type limits
NumPP SHALL provide `finfo(dtype)` for floating/complex dtypes exposing `eps`,
`tiny` (smallest normal), `max`, `min`, `resolution`, `bits`, and the mantissa/
exponent digit counts, and `iinfo(dtype)` for integer dtypes exposing `min`,
`max`, and `bits`, matching `numpy.finfo` / `numpy.iinfo`.

#### Scenario: limits match numpy
- WHEN `finfo(float64).eps` / `tiny` / `max` and `iinfo(int32).min` / `max` are read
- THEN each equals the corresponding `numpy.finfo` / `numpy.iinfo` attribute
- WHEN `finfo` is given an integer dtype (or `iinfo` a float dtype)
- THEN it raises value_error

