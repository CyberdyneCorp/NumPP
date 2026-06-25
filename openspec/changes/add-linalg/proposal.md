# Add linear algebra (Phase 4)

## Why

With ufuncs in place, Phase 4 adds `numpy.linalg` parity: solvers, decompositions,
and matrix products. Every routine has a portable pure-C++ fallback so it works on
mobile/no-LAPACK builds, with a LAPACK-accelerated path behind `NUMPP_WITH_LAPACK`.

## What changes

- **linalg** capability: products (`dot`, `vdot`, `inner`, `outer`, `trace`,
  `kron`); solvers (`solve`, `inv`, `det`, `slogdet`, `matrix_power`); and
  decompositions (`cholesky`, `qr`, `svd`, `eig`, `eigvals`, `eigh`, `eigvalsh`,
  `lstsq`, `pinv`, `matrix_rank`, `norm`).
- Pure-C++ kernels: LU with partial pivoting (solve/inv/det), Householder QR,
  one-sided Jacobi SVD, symmetric Jacobi eigensolver, Cholesky. Computation in
  double/complex<double>; output dtype follows numpy.linalg (float32/complex
  preserved, else float64).

## Reuse vs rewrite

- Matrix products route through the existing `matmul` dispatcher (BLAS/CPU).
- Decompositions are clean-room C++; LAPACK path added via a weak vtable later.

## Non-goals

- Stacked/batched linalg (>2D) beyond what numpy broadcasts — deferred.
- `tensorsolve`/`tensorinv`, `multi_dot`, `einsum` (separate work).
