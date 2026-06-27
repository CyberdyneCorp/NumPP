# Changelog

## Unreleased — NumPy-parity expansion & real GPU acceleration

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
