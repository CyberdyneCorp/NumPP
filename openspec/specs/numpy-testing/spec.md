# numpy-testing Specification

## Purpose
TBD - created by archiving change tier-c-partial. Update Purpose after archive.
## Requirements
### Requirement: Test assertion helpers
NumPP SHALL provide array_equal, array_equiv, assert_array_equal, assert_allclose,
assert_array_almost_equal and assert_array_less matching numpy.testing semantics
(the assert_* functions raise on mismatch and return on success).

#### Scenario: testing helpers match numpy semantics
- WHEN array_equal/array_equiv classify two arrays
- THEN the boolean equals numpy.array_equal / numpy.array_equiv
- WHEN an assert_* helper is given matching arrays it returns; given mismatched
  arrays (beyond tolerance) it raises value_error

