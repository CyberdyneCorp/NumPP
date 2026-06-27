# dtype-system Specification

## ADDED Requirements

### Requirement: Casting-rule helpers
NumPP SHALL provide `promote_types(a, b)` returning the smallest dtype both inputs
can safely cast to, and `min_scalar_type(value)` returning the smallest dtype that
can hold a scalar, matching numpy (with NEP-50 semantics already used by
`result_type`/`can_cast`/`common_type`).

#### Scenario: promotion and minimal scalar type match numpy
- WHEN `promote_types` is queried for representative dtype pairs (int/uint/float/complex/bool)
- THEN the result equals `numpy.promote_types`
- WHEN `min_scalar_type` is queried for representative integer and floating scalars
- THEN the result equals `numpy.min_scalar_type`
