# linalg Specification

## ADDED Requirements

### Requirement: Array-API 2023 linear-algebra names
NumPP SHALL provide the array-API standard aliases `matrix_transpose` (swap the
last two axes), `permute_dims` (general axis permutation), `vecdot` (contract a
named axis of two arrays), `vector_norm` and `matrix_norm`, matching the
`numpy.linalg` / array-API definitions including `axis`/`keepdims`/`ord` options.

#### Scenario: aliases match numpy results
- WHEN `matrix_transpose`/`permute_dims`/`vecdot`/`vector_norm`/`matrix_norm` are
  called on representative inputs (batched matrices, complex vectors, various `ord`)
- THEN each result equals the corresponding `numpy.linalg`/`numpy` function within
  tolerance
