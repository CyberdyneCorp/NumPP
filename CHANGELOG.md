# Changelog

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

### Known limitations
See README "Known limitations" — tracked as issues #3, #7, #8, #9, #11, #14.
