# linalg Specification

## ADDED Requirements

### Requirement: Batched (stacked) linear algebra
The linear-algebra routines SHALL operate over the last two axes of an N-D stack
of matrices, matching numpy.linalg: an input of shape `(..., n, n)` (or `(..., m, n)`
for the least-squares/decomposition family) is treated as a stack of matrices and
the operation is applied to each, preserving the leading batch dimensions in the
output. This SHALL apply to `solve`, `inv`, `det`, `slogdet`, `matrix_power`,
`cholesky`, `qr`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `svd`, `svdvals`, `pinv`,
`matrix_rank` and `lstsq`. The existing 2-D behavior SHALL be unchanged (a single
matrix is the zero-batch case).

#### Scenario: stacked inputs match numpy's per-matrix results
- GIVEN an array `A` of shape `(k, n, n)`
- WHEN `det(A)` / `inv(A)` / `solve(A, b)` / `cholesky(A)` / `eigvals(A)` / `svdvals(A)` are computed
- THEN the result has numpy's batched shape (`(k,)`, `(k,n,n)`, `(k,n)`, …) and each
  batch element equals the corresponding single-matrix result (validated against
  numpy directly, or by reconstruction for sign/order-ambiguous decompositions)

### Requirement: tensorsolve and tensorinv
NumPP SHALL provide `tensorsolve(a, b, axes)` and `tensorinv(a, ind)` matching
`numpy.linalg.tensorsolve` / `numpy.linalg.tensorinv`.

#### Scenario: tensor solve/inverse match numpy
- WHEN `tensorsolve(a, b)` solves `a x = b` over the appropriate reshaped axes
- THEN `x` equals `numpy.linalg.tensorsolve(a, b)` within tolerance
- WHEN `tensorinv(a, ind)` is computed
- THEN it equals `numpy.linalg.tensorinv(a, ind)` within tolerance
