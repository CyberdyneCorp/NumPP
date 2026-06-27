# memory-overlap Specification

## ADDED Requirements

### Requirement: Memory-overlap detection
NumPP SHALL provide `shares_memory(a, b)` and `may_share_memory(a, b)` that report
whether two arrays can share underlying buffer bytes, matching numpy. `shares_memory`
SHALL be exact for the common cases (same buffer with overlapping byte extents,
views, slices) and `may_share_memory` SHALL be a fast conservative bounds check.

#### Scenario: views share memory, independent arrays do not
- GIVEN an array `a` and a slice/transpose view `v` of it
- THEN `shares_memory(a, v)` and `may_share_memory(a, v)` are both true
- GIVEN an independent copy `c = a.copy()`
- THEN `shares_memory(a, c)` is false
- AND the booleans equal `numpy.shares_memory` / `numpy.may_share_memory`
