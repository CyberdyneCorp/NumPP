# NumPP roadmap progress (autonomous build loop)

Target: **full NumPy parity**, phase by phase. Mode: **fully autonomous**. Every
oracle-caught bug gets a GitHub issue + a regression test. Each phase is an
OpenSpec change → branch → PR → merge → archive.

Durable tracker (survives context resets). Update after every increment.

| Phase | Title | OpenSpec change | Status |
|------:|-------|-----------------|:------:|
| 0–2 | Foundation, ndarray, dtypes | bootstrap-numpp-foundation | ✅ merged+archived |
| 3 | ufuncs & element-wise math | add-ufuncs-elementwise | ✅ merged+archived |
| 4 | linalg (solve/inv/det/svd/qr/eig/cholesky/lstsq/norms) | add-linalg | ✅ merged+archived |
| 5 | fft | add-fft | ✅ merged+archived |
| 6 | random | add-random | ✅ merged+archived |
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
- [x] svd (Jacobi on Gram matrix; reduced+full), svdvals, pinv, matrix_rank, lstsq, norm 2/-2/nuc
- [x] eig/eigvals (general; complex Hessenberg QR + inverse iteration)
- [x] LAPACK path behind NUMPP_WITH_LAPACK (weak vtable; solve routed, portable fallback default)

Increment 1: 8 linalg test cases (67 total / 439 checks), 0 divergences,
clang+gcc+ASan green. Compute in double/complex<double>, output dtype per numpy.

## Phase 5 — fft: sub-tracker (branch phase-5-fft)

- [x] DFT engine: radix-2 Cooley-Tukey + Bluestein (arbitrary/prime n)
- [x] fft/ifft (axis, n pad/truncate, norm backward/ortho/forward); complex64/128 out
- [x] rfft/irfft, hfft/ihfft
- [x] fftfreq/rfftfreq/fftshift/ifftshift
- [x] fft2/ifft2/rfft2/irfft2, fftn/ifftn/rfftn/irfftn

Increment 1: 7 fft cases (95 total / 506 checks), 0 divergences, clang+gcc+ASan green.

## Phase 6 — random: sub-tracker (branch phase-6-random)

- [x] SeedSequence (exact) + PCG64 (XSL-RR); bit-exact raw stream + random doubles
- [x] Generator: integers (32-bit Lemire), uniform, shuffle/permutation (bit-exact); choice (replace=True bit-exact)
- [x] Distributions: standard_normal/exponential/normal/exponential/gamma/beta/chisquare/poisson/binomial (correct samplers; statistical parity, issue #8)
- [x] MT19937 (legacy seeding) + RandomState (random_sample/randint/randn/normal/uniform) bit-exact; standalone MT19937 raw -> issue #9

Increment 1: 5 cases (111 total / 553 checks), bit-exact vs numpy.random.PCG64,
0 divergences, clang+gcc+ASan green.

## Bug log (oracle divergences → GitHub issues)

| # | Issue | Phase | Regression test | Status |
|---|-------|------:|-----------------|--------|
| #2 | minimum/maximum didn't propagate NaN | 3 | test_ufunc2 "minimum/maximum propagate NaN" | fixed |
| #8 | distributions not bit-exact (no ziggurat tables) | 6 | test_random3 moments | open |
| #7 | choice(replace=False) not bit-exact w/ numpy | 6 | test_random2 valid-sample | open |
| #3 | Phase 3 parity tracking (out=/where=, nan-reductions, argmin/cumsum) | 3 | — | open |
