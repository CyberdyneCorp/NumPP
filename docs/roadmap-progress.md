# NumPP roadmap progress (autonomous build loop)

Target: **full NumPy parity**, phase by phase. Mode: **fully autonomous**. Every
oracle-caught bug gets a GitHub issue + a regression test. Each phase is an
OpenSpec change → branch → PR → merge → archive.

Durable tracker (survives context resets). Update after every increment.

| Phase | Title | OpenSpec change | Status |
|------:|-------|-----------------|:------:|
| 0–2 | Foundation, ndarray, dtypes | bootstrap-numpp-foundation | ✅ merged+archived |
| 3 | ufuncs & element-wise math | add-ufuncs-elementwise | ✅ merged+archived |
| 4 | linalg (solve/inv/det/svd/qr/eig/cholesky/lstsq/norms) | add-linalg | 🟡 in progress |
| 5 | fft | — | ⬜ |
| 6 | random | — | ⬜ |
| 7 | I/O (.npy/.npz, printing/repr) | — | ⬜ |
| 8 | sorting/searching/counting, set ops, unique | — | ⬜ |
| 9 | structured/record dtypes, datetime64, strings | — | ⬜ |
| 10 | GPU kernel coverage beyond GEMM | — | ⬜ |
| 11 | hardening, docs, v1.0 | — | ⬜ |

## Phase 3 — ufuncs: sub-tracker

Engine: broadcasting N-ary element-wise execution over the existing
ndarray/dtype/broadcast machinery, with per-ufunc type-resolution policy.

- [x] Core engine: broadcasted binary/unary loops, type policies (oracle-verified)
- [x] Arithmetic: add, subtract, multiply, true_divide, floor_divide, mod, power, negative, positive, absolute, sign
- [x] Comparison: equal, not_equal, less, less_equal, greater, greater_equal
- [x] Logical: logical_and/or/xor/not
- [x] Bitwise: and, or, xor, invert, left_shift, right_shift
- [x] Math: sqrt, cbrt, exp, expm1, log, log2, log10, log1p, square, reciprocal
- [x] Trig/hyperbolic: sin/cos/tan, arc*, sinh/cosh/tanh, arc*h, deg2rad, rad2deg, arctan2
- [x] Float ops: floor, ceil, trunc, rint, copysign, hypot, isnan/isinf/isfinite, signbit
- [x] minimum, maximum, fmin, fmax, clip
- [x] Reductions: sum, prod, min, max, mean, std, var (ddof), any, all (axis, keepdims, dtype)  · [ ] axis-tuple
- [x] where, nonzero
- [x] Operators: + - * /, unary -, comparison operators, weak-promoted scalar operands
- [x] complex components: conj/conjugate/real/imag/angle
- [x] out=/where= via copyto + arithmetic overloads; integer**negative raises (parity)
- [ ] long-tail (issue #3): out= on ALL ufuncs, nan-reductions, argmin/argmax, cumsum/cumprod, axis-tuple

Increment 1: core families. Increment 2: trig/math extras, predicates,
binary-float, shifts, fmin/fmax, clip, var/std, where, nonzero, scalars.
Increment 3: conj/real/imag/angle, copyto + out=/where=, integer-power error.
**60 cases / 418 checks, clang+gcc+ASan green, 0 oracle divergences.** Phase 3
complete to a high parity bar; genuine long-tail tracked in #3.

## Phase 4 — linalg: sub-tracker (branch phase-4-linalg)

- [x] Products: dot, vdot, inner, outer, trace, kron
- [x] LU core: solve, inv, det, slogdet, matrix_power (real + complex)
- [x] cholesky (+ not-PD -> linalg_error)
- [x] qr (Householder, reduced/complete), eigh/eigvalsh (Jacobi; complex via 2n embedding), norm (vector + matrix 1/inf/fro)
- [ ] svd (Jacobi), svdvals
- [ ] eig/eigvals (general), lstsq, pinv, matrix_rank, norm
- [ ] LAPACK path behind NUMPP_WITH_LAPACK (weak vtable)

Increment 1: 8 linalg test cases (67 total / 439 checks), 0 divergences,
clang+gcc+ASan green. Compute in double/complex<double>, output dtype per numpy.

## Bug log (oracle divergences → GitHub issues)

| # | Issue | Phase | Regression test | Status |
|---|-------|------:|-----------------|--------|
| #2 | minimum/maximum didn't propagate NaN | 3 | test_ufunc2 "minimum/maximum propagate NaN" | fixed |
| #3 | Phase 3 parity tracking (out=/where=, nan-reductions, argmin/cumsum) | 3 | — | open |
