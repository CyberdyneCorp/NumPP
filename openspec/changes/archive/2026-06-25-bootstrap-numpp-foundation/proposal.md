# Bootstrap NumPP foundation

## Why

NumPP is a clean-room C++20 port of NumPy, the sibling of SymPP (which ports
SymPy). NumPy is enormous, so we commit to a **phased roadmap** and detail only
the foundation now: the project skeleton, the N-D array core, the dtype system,
the NumPy validation oracle, and — the one thing we deliberately do differently
from NumPy — a **tiered acceleration layer** that adds BLAS/LAPACK and GPU
(Metal/Vulkan/CUDA/OpenCL) when available but always falls back to a portable,
dependency-free C++ CPU kernel so the library still compiles and runs on iOS and
Android hardware that has neither accelerated linear algebra nor a usable GPU.

This change establishes Phase 0 (foundation + oracle + build) and Phase 1
(ndarray core + dtypes) and locks in the backend-dispatch architecture that
every later phase depends on. Everything else (ufuncs, full linalg, fft, random,
etc.) follows as separate OpenSpec changes against this baseline.

## What changes

This change introduces the following capabilities (spec deltas):

- **ndarray-core** — the `numpp::ndarray` N-dimensional, strided, reference-
  counted buffer container: shape/strides/offset, contiguous and non-contiguous
  views, copy-vs-view semantics, reshape/transpose/squeeze, basic and sliced
  indexing, broadcasting rules, array-creation routines (zeros/arange/linspace/
  eye/...), flatten/ravel, dimension insertion, writeability/read-only views,
  and raw data access.
- **error-handling** — the exception model: a `numpp::error` base and a typed
  hierarchy (`value_error`, `type_error`, `index_error`, `axis_error`,
  `linalg_error`, `not_implemented_error`) mirroring NumPy's error categories,
  with strong exception safety and a documented throw-vs-propagate contract.
- **dtype-system** — the dtype enumeration (bool, int8..int64, uint8..uint64,
  float16/32/64, complex64/128), element sizes, type-promotion/result-type
  rules, and safe/unsafe casting — matching NumPy semantics.
- **backend-dispatch** — the tiered runtime acceleration layer: a portable CPU
  kernel that is always present, optional weak-linked BLAS/LAPACK and GPU
  backends gated by compile-time feature flags, runtime device + size-threshold
  selection, and a guaranteed CPU fallback. Defines the mobile (no-accel) path.
- **oracle-testing** — the NumPy validation harness: tests express expected
  results by running real Python NumPy and asserting `allclose`, with a checked
  generation mode so the comparison data can be frozen for CI without Python.
- **build-packaging** — CMake (>= 3.25) C++20 project layout, feature flags,
  vcpkg + Conan packaging, and the iOS/Android cross-compilation contract.

## Phased roadmap (scope guardrail)

Only Phases 0–1 (plus the cross-cutting backend layer) are specified in detail
in this change. Later phases are listed here as the agreed roadmap and will each
arrive as their own OpenSpec change. This is the scope boundary: **do not
implement Phase 2+ in this change.**

| Phase | Title | This change? |
|------:|-------|:---:|
| 0 | Foundation, build/packaging, NumPy oracle harness, backend-dispatch architecture | ✅ |
| 1 | `ndarray` core: storage, views, strides, reshape/transpose, indexing/slicing, broadcasting | ✅ |
| 2 | dtype system: dtypes, promotion, casting | ✅ (foundation subset) |
| 3 | ufuncs & element-wise math: arithmetic, comparison, trig/exp/log, reductions, `where`, broadcasting execution | ⬜ later |
| 4 | linalg: `matmul/dot`, `solve`, `inv`, `det`, `svd`, `qr`, `eig`, `cholesky`, `lstsq`, norms (BLAS/LAPACK accelerated) | ⬜ later |
| 5 | `fft` module | ⬜ later |
| 6 | `random` module (Generator/BitGenerator) | ⬜ later |
| 7 | I/O: `.npy`/`.npz` save/load, array printing/`repr` | ⬜ later |
| 8 | Sorting/searching/counting, set ops, `unique` | ⬜ later |
| 9 | Structured/record dtypes, datetime64, strings | ⬜ later |
| 10 | GPU kernel coverage beyond GEMM (element-wise/reductions on device) | ⬜ later |
| 11 | Hardening, docs, v1.0 packaging | ⬜ later |

## Reuse vs rewrite

Per project policy, prefer adapting proven code over reinventing it — after
testing it:

- **Adapt from matlab_llvm runtime** (`/home/leonardo/work/matlab_llvm/runtime`):
  the dense descriptor layout, the RAII `MatPtr`/custom-deleter pattern, the
  `shape_op<IndexFn>` template for transpose/reshape, the BLAS dispatch with
  row/col-major handling and size thresholds, and the **weak-symbol GPU tiering**
  (`gpu/runtime_gpu.cpp`, `MATLAB_GPU_TARGET=cpu|metal|cuda|opencl|auto`) — this
  is exactly the fallback model we want.
- **Adapt from NumPy** (`/home/leonardo/work/numpy/numpy/_core/src`): the
  half-precision math (`common/half.hpp`, `npymath/halffloat.cpp`), bit/util
  helpers (`common/utils.hpp`), float-status RAII (`common/float_status.hpp`),
  the portable SIMD intrinsics (`common/simd`), and the dtype-promotion *rules*
  (re-expressed without `PyObject`). Sorting templates (`npysort/*.hpp`) for
  Phase 8.
- **Mirror from SymPP**: project layout, CMake options, vcpkg/Conan packaging,
  and the oracle-against-the-Python-original testing methodology.
- **Rewrite**: anything touching the CPython C-API (`PyObject*`, `PyArray_*`,
  ufunc Python dispatch) — re-implemented as pure C++.

## Non-goals

- **Not** a Python extension module. No CPython C-API, no `PyObject`. (Optional
  Python bindings may come much later as a separate change.)
- **Not** drop-in ABI/source compatibility with NumPy's C-API.
- **No** Phase 2+ feature implementation in this change (ufunc library, full
  linalg, fft, random, I/O are roadmap only).
- **No** object/`dtype=object` arrays, masked arrays, or `np.matrix`.
- **No** distributed/out-of-core arrays.
- GPU backends beyond **GEMM/matmul** dispatch are out of scope here (Phase 10);
  this change only establishes the dispatch *architecture* and the CPU fallback.
- Not committing to NumPy's exact memory ABI or pickle format.
