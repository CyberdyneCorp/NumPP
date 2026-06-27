# error-state Specification

## ADDED Requirements

### Requirement: Floating-point error-state control
NumPP SHALL provide `seterr`/`geterr` to read and set how floating-point
conditions (divide, over, under, invalid) are treated, and an `errstate` scoped
guard that restores the prior state on destruction, matching `numpy.errstate`
semantics for the modes `ignore`, `warn`, `raise`, and `call`.

#### Scenario: errstate scopes the policy and restores it
- GIVEN the default error state
- WHEN code runs inside an `errstate(divide="raise")` guard and divides by zero
- THEN a floating-point error is raised
- AND after the guard exits the previous state is restored
- AND `geterr()` reflects the same keys/values numpy reports
