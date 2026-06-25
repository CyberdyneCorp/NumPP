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
- [x] Bitwise: and, or, xor, invert  · [ ] left_shift, right_shift
- [x] Math: sqrt, exp, log, square, reciprocal  · [ ] cbrt, expm1, log2, log10, log1p
- [x] Trig: sin, cos, tan  · [ ] arc*/hyperbolic/arctan2/deg2rad/rad2deg
- [x] floor, ceil  · [ ] trunc, rint, copysign, hypot, isnan/isinf/isfinite, signbit
- [x] minimum, maximum  · [ ] fmin, fmax, clip
- [x] Reductions: sum, prod, min, max, mean, any, all (axis, keepdims, dtype)  · [ ] std, var, axis-tuple
- [ ] where, nonzero
- [x] Operators: + - * / and unary -  · [ ] comparison operators, scalar operands
- [ ] kwargs parity: out=, where=

Increment 1 committed: 11 ufunc test cases, **0 oracle divergences** (type rules
pre-derived from NumPy). Phase 3 ~60% — extras above remain before PR/merge.

## Bug log (oracle divergences → GitHub issues)

| # | Issue | Phase | Regression test | Status |
|---|-------|------:|-----------------|--------|
| – | (none yet) | | | |
