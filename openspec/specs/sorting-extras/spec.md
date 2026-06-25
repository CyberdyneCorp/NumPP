# sorting-extras Specification

## Purpose
TBD - created by archiving change add-sorting-extras. Update Purpose after archive.
## Requirements
### Requirement: Indirect and complex sorting
NumPP SHALL provide lexsort (stable multi-key indirect sort, last key primary),
sort_complex (real then imaginary ordering) and a searchsorted overload taking a
`sorter` index array, matching numpy.

#### Scenario: lexsort matches numpy
- WHEN lexsort is applied to a sequence of keys
- THEN the index order equals numpy.lexsort (last key primary, stable)

#### Scenario: sort_complex and sorter-based searchsorted match numpy
- WHEN sort_complex is applied to a complex (or real) array
- THEN the result equals numpy.sort_complex
- WHEN searchsorted is given an unsorted array and its sorter
- THEN the insertion indices equal numpy.searchsorted(..., sorter=...)

