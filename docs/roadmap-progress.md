# NumPP roadmap progress (autonomous build loop)

Target: **full NumPy parity**, phase by phase. Mode: **fully autonomous**. Every
oracle-caught bug gets a GitHub issue + a regression test. Each phase is an
OpenSpec change → branch → PR → merge → archive.

Durable tracker (survives context resets). Update after every increment.

| Phase | Title | OpenSpec change | Status |
|------:|-------|-----------------|:------:|
| 0–2 | Foundation, ndarray, dtypes | bootstrap-numpp-foundation | ✅ merged+archived |
| 3 | ufuncs & element-wise math | add-ufuncs-elementwise | 🟡 in progress |
| 4 | linalg (solve/inv/det/svd/qr/eig/cholesky/lstsq/norms) | — | ⬜ |
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
- [ ] kwargs parity: out=, where= (tracked in #3)

Increment 1: core families, 0 divergences. Increment 2: trig/math extras,
predicates, binary-float, shifts, fmin/fmax, clip, var/std, where, nonzero,
scalar operands — 56 cases / 404 checks, clang+gcc+ASan green. Phase 3 ~90%;
only out=/where= kwargs + axis-tuple remain (issue #3) before PR/merge.

## Bug log (oracle divergences → GitHub issues)

| # | Issue | Phase | Regression test | Status |
|---|-------|------:|-----------------|--------|
| #2 | minimum/maximum didn't propagate NaN | 3 | test_ufunc2 "minimum/maximum propagate NaN" | fixed |
| #3 | Phase 3 parity tracking (out=/where=, nan-reductions, argmin/cumsum) | 3 | — | open |
