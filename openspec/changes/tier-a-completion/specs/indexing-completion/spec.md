# indexing-completion Specification

## ADDED Requirements

### Requirement: Fancy and boolean indexing
NumPP SHALL provide boolean-mask and integer-array (fancy) get and set over an
array, matching numpy's `a[mask]`, `a[mask]=v`, `a.flat[idx]`, `a.flat[idx]=v`.

#### Scenario: fancy/boolean access matches numpy
- WHEN boolean_index/boolean_assign/fancy_index/fancy_assign are applied
- THEN the result equals numpy boolean / integer-array indexing

### Requirement: Index toolkit helpers
NumPP SHALL provide put_along_axis, place, ix_, fill_diagonal, diag_indices,
tril_indices and triu_indices matching numpy.

#### Scenario: index helpers match numpy
- WHEN put_along_axis/place/ix_/fill_diagonal/diag_indices/tril_indices/
  triu_indices are evaluated
- THEN the result equals the corresponding numpy function
