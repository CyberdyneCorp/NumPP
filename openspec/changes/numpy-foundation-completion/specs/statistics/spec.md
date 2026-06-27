# statistics Specification

## ADDED Requirements

### Requirement: Trapezoidal integration
NumPP SHALL provide `trapz` (and the `trapezoid` alias) computing the composite
trapezoidal integral along an axis, accepting either uniform spacing `dx` or
explicit sample points `x`, matching `numpy.trapezoid`.

#### Scenario: trapezoidal rule matches numpy
- WHEN `trapz(y)` / `trapz(y, dx=h)` / `trapz(y, x=xs)` is evaluated (1-D and along an axis of an N-D array)
- THEN the result equals `numpy.trapezoid` with the same arguments within tolerance
