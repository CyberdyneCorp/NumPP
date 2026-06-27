# linalg Specification

## Purpose
TBD - created by archiving change add-linalg. Update Purpose after archive.
## Requirements
### Requirement: Matrix and vector products

NumPP SHALL provide `dot`, `vdot`, `inner`, `outer`, `trace`, and `kron` matching
NumPy, routing 2-D matrix products through the backend dispatcher.

#### Scenario: Matrix product
- WHEN `dot` is applied to two 2-D arrays
- THEN the result equals `numpy.dot` of the same inputs

#### Scenario: Outer product
- WHEN `outer` is applied to two 1-D arrays of length m and n
- THEN the result has shape `[m, n]` and equals `numpy.outer`

### Requirement: Linear solvers

NumPP SHALL provide `solve`, `inv`, `det`, `slogdet`, and `matrix_power` matching
`numpy.linalg`, for real and complex matrices, using LU decomposition with partial
pivoting in the portable path. A singular matrix SHALL raise `linalg_error`.

#### Scenario: Solve a linear system
- GIVEN a non-singular matrix A and vector b
- WHEN `solve(A, b)` is computed
- THEN `A @ x` equals `b` within tolerance and `x` equals `numpy.linalg.solve(A, b)`

#### Scenario: Determinant
- WHEN `det(A)` is computed
- THEN it equals `numpy.linalg.det(A)` within tolerance

#### Scenario: Singular matrix
- WHEN `inv` is applied to a singular matrix
- THEN a `linalg_error` is raised

### Requirement: Decompositions

NumPP SHALL provide `cholesky`, `qr`, `svd`, `eig`, `eigvals`, `eigh`,
`eigvalsh`, `lstsq`, `pinv`, `matrix_rank`, and `norm` matching `numpy.linalg`.
Because factor signs/orderings are not unique, conformance SHALL be checked by
reconstruction (e.g. `A = Q@R`, `A = U@diag(S)@Vh`, `A@v = λ@v`) and by comparing
quantities that are unique (singular values, sorted eigenvalues).

#### Scenario: Cholesky reconstruction
- GIVEN a positive-definite matrix A
- WHEN `cholesky(A)` returns L
- THEN `L @ L.conj().T` equals A within tolerance, and a non-positive-definite
  input raises `linalg_error`

#### Scenario: QR reconstruction
- WHEN `qr(A)` returns Q, R
- THEN `Q @ R` equals A within tolerance and Q has orthonormal columns

#### Scenario: SVD singular values
- WHEN `svd(A)` returns U, S, Vh
- THEN S equals `numpy.linalg.svd(A)[1]` within tolerance and `U@diag(S)@Vh` reconstructs A

#### Scenario: Symmetric eigenvalues
- WHEN `eigvalsh(A)` is computed for a symmetric A
- THEN the sorted result equals `numpy.linalg.eigvalsh(A)` within tolerance

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

