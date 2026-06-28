# Changelog

## 1.3.7 — 2026-06-27 — test_numeric/einsum/function_base/polynomial mining + roots() complex-root fix

Continues mining NumPy's own test suites through the live-NumPy oracle across the
array-math, einsum, lib and polynomial layers. **904 cases / 2415 checks, 0
divergences.** One real bug found and fixed:

- **`roots()` no longer drops the imaginary part of complex roots.** It built its
  result from `real(eigvals(companion))`, so a polynomial with complex-conjugate
  roots (e.g. `x^2 + 1`) returned `[0, 0]` instead of `±i`. Both real and imaginary
  parts of the companion-matrix eigenvalues are now preserved; the result is a real
  array only when every root is real (matching `numpy.roots`), `complex128`
  otherwise. All-real-root polynomials are unaffected. (#104)

New oracle-mined regression coverage (no further divergences):

- **`test_numeric.py`** — `cross` (3-D, 2-D z-component, batched rows), `tensordot`
  (single-axis and explicit axis lists), `outer`/`inner`, `correlate`/`convolve`
  (full/same/valid).
- **`test_einsum.py`** — trace/diagonal/transpose/reductions, matmul/dot/outer,
  implicit output, ellipsis `...ij->...ji`, batched `bij,bjk->bik`, chained
  three-operand `ij,jk,kl->il`.
- **`test_function_base.py`** — `interp` (basic, left/right fills, periodic), `trapz`
  (dx and explicit x), `bincount` (counts, weights, minlength), `percentile`/`median`
  (incl. q=0/100), 1-D `gradient`, `unwrap`.
- **`test_polynomial.py`** — `roots` (real, complex, cubic), `polyval`, `polyfit`
  (linear/quadratic), `polyder`/`polyint`, `poly`, `polyadd`/`polymul`, `vander`.

## 1.3.6 — 2026-06-27 — test_nanfunctions.py mining: NaN-reduction fixes

Mines NumPy's own `numpy/lib/tests/test_nanfunctions.py` through the live-NumPy
oracle (all-NaN slices, partial-NaN data, per-axis mixes, ddof edges). **879 cases /
2359 checks, 0 divergences.** Three real bugs found and fixed:

- **`nanmax`/`nanmin` of an all-NaN slice now return `nan`** (was `±inf`). The
  kernels fill NaN with `∓inf` then reduce, so an all-NaN slice collapsed to the
  fill value; fixed by setting zero-count slices to `nan` (scalar + per-axis). (#94)
- **`nanargmin`/`nanargmax` of an all-NaN slice now raise `value_error`** (was a
  silent bogus index `0`), matching numpy's `ValueError`. (#96)
- **`nanvar`/`nanstd` with `ddof >= non-NaN count` now return `nan`** (was `inf` or
  negative), matching numpy's degrees-of-freedom ≤ 0 rule. (#97)

## 1.3.5 — 2026-06-27 — INT_MIN crash fix + test_multiarray.py mining

Continues mining NumPy's own tests through the live-NumPy oracle (now `test_umath.py`
overflow cases and `test_multiarray.py` array-core behaviors). **862 cases / 2313
checks, 0 divergences.**

- **Crash fix: `floor_divide`/`remainder` of `INT_MIN` by `-1`** aborted the process
  with SIGFPE (signed-overflow → hardware divide exception; UB in C++). numpy wraps
  to `INT_MIN`. Guarded the `MIN/-1` case in `int_floordiv` (→ `INT_MIN`) and factored
  remainder into `int_mod` (→ `0`, since `x % -1 == 0`). (#87)
- **Array-core regression coverage** from `test_multiarray.py`: argmax/argmin
  (NaN-wins), negative-step slicing, clip/diagonal/repeat/take, concatenate/stack,
  broadcast/transpose, fancy/boolean indexing, put/place/choose/compress,
  ravel/unravel_index, 0-d arrays, reduction dtype upcasting, integer overflow,
  astype casting, partition/argpartition, searchsorted, axis sort — all match numpy.

## 1.3.4 — 2026-06-27 — test_umath.py mining: ufunc special-value fixes

Mines NumPy's own `numpy/_core/tests/test_umath.py` by replaying special-value
input sets (±inf, nan, ±0) through the ufuncs against the live-NumPy oracle —
unary/binary special values, integer division/remainder (signs + by-zero),
`float_power`, bitwise ufuncs, NaN comparisons, complex branch cuts (generic
off-cut inputs), and real inverse-function domain edges. **824 cases / 2243 checks,
0 divergences.** Two real bugs found and fixed:

- **`sign(nan)` now returns `nan`** (was `0`). The real-floating kernel computed
  `(x>0)-(x<0)`, which is `0` for NaN; numpy returns `nan`. (#81)
- **Integer remainder by zero now returns `0`** (was the dividend). `Mod` computed
  `x - floordiv(x,0)*0 = x`; numpy returns `0`. `floor_divide` already matched. (#83)

(Complex transcendentals on branch cuts / signed zeros are implementation-defined —
numpy's npymath and platform libc++ disagree — so those exact points are excluded,
matching numpy's own platform xfails.)

## 1.3.2 — 2026-06-27 — cond(p) + comprehensive test_linalg.py mining

Completes mining of NumPy's own `numpy/linalg/tests/test_linalg.py` through the
live-NumPy oracle (34 added cases across solve/inv/det/slogdet/svd/pinv/matrix_rank/
cond/lstsq/eig/eigh/matrix_power/qr/norm/multi_dot, for real & complex, non-square,
size-0, 1×1 and stacked inputs). **804 cases / 2162 checks, 0 divergences.**

- **`cond(a, p)` — full numpy parity.** The mining showed `cond` only computed the
  2-norm. Added the `p` selector ∈ {1, -1, 2, -2, +inf, -inf, 'fro'}: 2/-2 use
  singular values, the rest use `norm(A,p)·norm(inv(A),p)`; a singular matrix yields
  infinity (the 2-norm path now returns inf instead of nan when σ_min is 0).
- Comprehensive linalg regression coverage from numpy's seed arrays/edge cases.

## 1.3.1 — 2026-06-27 — SVD accuracy fix + NumPy-test mining

Validation now also **mines NumPy's own test suite** (`numpy/linalg/tests/test_linalg.py`):
its seed arrays and edge cases (complex, non-square, size-0, 1×1, stacked) are replayed
through the live-NumPy oracle. **789 cases / 2118 checks, 0 divergences.**

- **SVD high relative accuracy (#74).** The SVD was computed from the eigendecomposition
  of `AᴴA`, which squares the condition number and left the smallest singular values
  accurate only to ~`√eps` (≈1e-8). A rank-deficient matrix was therefore reported as
  full rank, and `pinv`/`cond` lost accuracy near singularity. Replaced with a
  **one-sided Jacobi (Hestenes) SVD** for both real and complex (complex pairs are
  phase-aligned before the rotation), unified behind one `svd_jacobi<T>`. Small
  singular values now resolve to near machine zero and `matrix_rank`/`pinv` match
  numpy for near-singular inputs.

## 1.3.0 — 2026-06-27 — NumPy-foundation completion (base for a C++ SciPy port)

Closes the `numpy-foundation-completion` OpenSpec change — the remaining genuine
`numpy.*` primitives a future C++ SciPy port ("SciPP") will stand on. Boundary rule:
`numpy.*` belongs in NumPP, `scipy.*` belongs in SciPP. Nine features, each its own
oracle-validated PR merged after CI green. **770 oracle cases / 2083 checks vs NumPy
2.1.3, 0 divergences**; clang + gcc, clean under ASan/UBSan.

- **finfo / iinfo** — machine limits (eps/tiny/max/min/resolution/bits/mantissa/
  exponent) for float16/32/64 + complex, and integer-type bounds.
- **isclose / isposinf / isneginf** — the elementwise predicates (array form of
  `allclose`).
- **trapz / trapezoid** — composite trapezoidal integration (`dx=`/`x=`/`axis`).
- **promote_types / min_scalar_type** — complete the dtype casting-rule set.
- **batched / stacked linalg** — `solve`/`inv`/`det`/`slogdet`/`matrix_power`/
  `cholesky`/`qr`/`eig`/`eigh`/`eigvals`/`eigvalsh`/`svd`/`svdvals`/`pinv`/
  `matrix_rank` now operate over the last two axes of an N-D stack (numpy semantics);
  `lstsq` stays 2-D, matching numpy.
- **tensorsolve / tensorinv** — the two missing `numpy.linalg` tensor routines.
- **einsum ellipsis (`...`)** — broadcasting subscripts (batched matmul/diagonal),
  composing with `optimize=`/`einsum_path`.
- **interp left/right/period** — fill values and periodic interpolation.

## 1.2.0 — 2026-06-27 — Bucket C: portable NumPy long-tail

Closes the `add-portable-numpy-gaps` OpenSpec change — the set of NumPy features
that were both still missing *and* implementable in dependency-free C++ (Buckets A
and B remain deferred by design, as they need a Python runtime/object model or an
external dependency). Nine features, each shipped as its own oracle-validated PR
merged after CI green. **711 oracle cases / 1979 checks vs NumPy 2.1.3, 0
divergences**, on clang + gcc, clean under ASan/UBSan; CI green across
ubuntu/macos × clang/gcc + iOS/Android no-accel + sanitizers.

- **errstate / seterr / geterr** — floating-point error-state control
  (divide/over/under/invalid → ignore/warn/raise/call), a scoped `errstate` guard,
  `seterrcall`, and a `floating_point_error`; the binary/unary float kernels clear
  and test the host FP flags and apply the active policy.
- **shares_memory / may_share_memory** — bounds-based memory-overlap detection.
- **ndindex / ndenumerate / nditer** — public C-order iteration API (correct for
  non-contiguous/transposed/broadcasted views).
- **array-API 2023 linalg aliases** — `matrix_transpose`, `permute_dims`,
  `vecdot`, `vector_norm`, `matrix_norm`.
- **busdaycalendar** — custom business-day calendars (`weekmask`/`holidays`)
  threaded through `is_busday`/`busday_count`/`busday_offset`.
- **einsum `optimize=` / `einsum_path`** — greedy pairwise contraction order.
- **masked hard/soft masks** — `harden_mask`/`soften_mask`/`hardmask` and
  mask-aware item assignment on `numpy.ma` MaskedArray.
- **polynomial domain/window + fit** — domain→window mapping and class-level
  `fit` on Chebyshev/Legendre/Hermite/HermiteE/Laguerre.
- **variable-length StringDType** — a numpy-2.0-style variable-length UTF-8 string
  array (standalone container; full ndarray-dtype integration deferred).

## 1.1.0 — 2026-06-27 — NumPy-parity expansion & real GPU acceleration

Builds on 1.0.0 to cover the full practical NumPy surface and add real GPU
backends. **1861 oracle checks across 649 cases vs NumPy 2.1.3, zero
divergences**, on clang + gcc, clean under ASan/UBSan; CI green across
ubuntu/macos × clang/gcc + iOS/Android no-accel + sanitizers.

- **Array manipulation, statistics, grids, indexing** — concatenate/stack/split/
  tile/repeat/flip/roll/rot90/pad/atleast_*/block/dsplit/trim_zeros; cumsum/diff/
  gradient/ptp/median/percentile/quantile/cov/corrcoef/digitize + nan-variants,
  histogram2d/histogramdd; meshgrid/indices/diag/tri*/vander/logspace/geomspace/
  mgrid/ogrid/fromiter/frombuffer; take/take_along_axis/put/diagonal/argwhere/
  compress/choose/select/ix_/fill_diagonal/diag_indices and **fancy + boolean
  ndarray indexing** (get & set).
- **einsum & tensors** — `einsum` with a general subscript parser, plus
  tensordot/cross/cond/multi_dot.
- **Polynomials** — convolve/correlate/interp; legacy `poly*` + `poly1d`; the
  `numpy.polynomial` package (power/Chebyshev/Legendre/Hermite/HermiteE/Laguerre
  val/vander/roots/der/int + classes).
- **ufunc & misc** — around/fix/gcd/lcm/sinc/degrees/nan_to_num/logaddexp/
  float_power/modf/frexp/ldexp/divmod/unwrap/i0/nextafter/spacing, type-check
  helpers, `emath`, packbits/unpackbits; sorting extras (lexsort/sort_complex/
  searchsorted-sorter); stride tricks (sliding_window_view/as_strided/piecewise/
  apply_along_axis/vectorize).
- **Text/char I/O & printing** — loadtxt/savetxt/genfromtxt, fromstring/tofile/
  fromfile, binary_repr/base_repr, `numpy.char` string ops, set_printoptions/
  array2string/format_float_*.
- **datetime / testing / masked arrays** — business-day (is_busday/busday_count/
  busday_offset)/datetime_as_string; `numpy.testing` asserts; `numpy.ma`
  MaskedArray with constructors, arithmetic, per-axis reductions, getmask/getdata.
- **Bit-exact random** — SFC64, **Philox** and standalone **MT19937**
  BitGenerators; the **ziggurat** `standard_normal`/`standard_exponential` (and
  hence `normal`) and `choice(replace=False)` are now bit-exact with numpy
  (reverse-engineered from numpy's source — Philox `M0 = 0xD2E7470EE14C6C93` +
  increment-before-generate; MT19937 generate_state(624)/`mt[0]=0x80000000`/
  `pos=623`; verbatim ziggurat tables). 20+ further distributions added
  (statistically validated). **Fixes #7, #8, #9, #36.**
- **Real GPU backends** — OpenCL and CUDA behind the weak-vtable slots:
  float32/float64 arithmetic/sqrt, reductions and **matmul** on the device, a
  device-buffer reuse pool and a 16×16 tiled shared-memory GEMM (~5–8× the CPU at
  fp64). Validated on an NVIDIA RTX 5060 (CUDA uses a PTX-virtual arch so an older
  nvcc JITs to a newer GPU). Default build stays pure-CPU and dependency-free.
- **Interop** — DLPack (`to_dlpack`/`from_dlpack`, zero-copy) and `memmap`
  (mmap-backed arrays).
- **Fixes** — #26 (spacing sign for negatives); a CI build fix (missing
  `<algorithm>` include).

## 1.0.0

First release: a C++20 port of NumPy, built phase by phase and validated against
real NumPy 2.1.3 as a numerical oracle (960 oracle checks, zero divergences;
clang + gcc, clean under ASan/UBSan).

### Phases

- **0–2 Foundation** — `ndarray` (N-D strided, shared-buffer views, reshape/
  transpose/squeeze/broadcast, basic+sliced indexing), dtype system (bool/int/
  uint/float16-64/complex64-128, NEP-50 promotion, casting), `numpp::error`
  hierarchy, NumPy-oracle test harness, tiered backend-dispatch architecture,
  CMake/vcpkg/Conan packaging, CI (clang/gcc, ASan, iOS/Android no-accel guard).
- **3 ufuncs** — arithmetic/comparison/logical/bitwise/shift, full unary math +
  trig/hyperbolic, predicates, reductions, where/clip/nonzero, operators with
  NEP-50 weak scalar promotion, out=/where=. Fixed #2 (minimum/maximum NaN).
- **4 linalg** — products, LU solvers, cholesky, qr, eigh/eigvalsh, svd family
  (pinv/matrix_rank/lstsq), general eig/eigvals, all norms; optional LAPACK path.
- **5 fft** — Cooley-Tukey + Bluestein DFT core; fft/ifft, rfft/irfft, hfft/ihfft,
  2-D and N-D variants, fftfreq/fftshift.
- **6 random** — SeedSequence + PCG64 + MT19937; Generator + legacy RandomState
  bit-exact with numpy; distributions (statistically validated, #8).
- **7 I/O** — `.npy`/`.npz` save/load (numpy-interop), numpy-compatible printing.
- **8 sorting** — sort/argsort/partition, searchsorted, unique, set ops,
  bincount, histogram, argmin/argmax, count_nonzero.
- **9 extended dtypes** — strings ('U'/'S'), datetime64/timedelta64, structured
  dtypes; core DType extended with a metadata side-channel (numeric dtypes
  unaffected).
- **10 device kernels** — weak-vtable GPU dispatch for ufuncs/reductions with a
  CPU-reference device backend proving the path end-to-end.
- **11 hardening** — nan-reductions (nansum/nanmean/nanmin/nanmax/nanvar/nanstd),
  complexity refactor of the printer, comprehensive docs, v1.0.0.

### Known limitations (as of 1.0.0)
Tracked as issues #3, #7, #8, #9, #11, #14. (Post-1.0, #7/#8/#9 — and the later
Philox #36 — are now fixed; see the Unreleased section. Remaining deferred items
and rationale are in `docs/numpy-parity-gaps.md`.)
