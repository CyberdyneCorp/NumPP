# Tasks — Phase 4 linalg

- [x] 1 Products: dot, vdot, inner, outer, trace, kron (route 2-D through matmul)
- [x] 2 LU core (partial pivoting); solve, inv, det, slogdet, matrix_power (real + complex)
- [x] 3 cholesky (+ not-positive-definite -> linalg_error)
- [x] 4 qr (Householder) — reconstruction-tested
- [x] 5 svd (one-sided Jacobi) — singular values + reconstruction
- [x] 6 eigh/eigvalsh (symmetric Jacobi)
- [x] 7 eig/eigvals (general; QR iteration or companion) — reconstruction A v = λ v
- [x] 8 lstsq, pinv, matrix_rank (via svd)
- [x] 9 norm (vector 1/2/inf/-inf/p; matrix 'fro'/1/2/inf/nuc)
- [x] 10 LAPACK-accelerated path behind NUMPP_WITH_LAPACK (weak lapack vtable)
- [x] 11 NumPy-oracle tests for every routine; issue + regression per divergence
- [x] 12 openspec validate --strict; clang+gcc + ASan/UBSan; PR + merge + archive
