# Design — NumPP foundation

## Context

NumPP ports NumPy to C++20. The hard constraint that shapes the architecture is
**portability**: the library must compile and run on iOS and Android, where
there is frequently no system BLAS/LAPACK and no GPU compute path we can rely on.
Therefore the *only* mandatory implementation of every numeric operation is a
portable, dependency-free C++ kernel; acceleration is strictly additive and
optional. This is the same model the matlab_llvm runtime uses (weak-linked
backends, `MATLAB_GPU_TARGET`, CPU fallback) and we adopt it deliberately.

## Goals / non-goals

- Goal: one source tree that builds three ways from the same code — mobile
  (CPU-only), desktop (+BLAS/LAPACK), server/workstation (+GPU) — selecting the
  fastest available path at runtime within a single binary when backends are
  compiled in.
- Goal: every result is validatable against real NumPy.
- Non-goal (this change): the ufunc library, full linalg, fft, random. Only the
  array core, dtypes, the dispatch skeleton, oracle, and build are in scope.

## Module / repo layout (mirrors SymPP)

```
include/numpp/
  numpp.hpp            # umbrella header
  fwd.hpp              # forward decls
  version.hpp(.in)
  core/                # ndarray, dtype, shape/strides, broadcasting, iterators
  backend/             # dispatch, capability registry, cpu kernels, blas/, gpu/
  linalg/  fft/  random/  io/   # (later phases; dirs reserved)
src/<module>/          # implementation .cpp mirroring include/
tests/                 # Catch2 + NumPy oracle harness
cmake/  conanfile.py  vcpkg.json  CMakeLists.txt
```

## ndarray memory model

`numpp::ndarray` is a non-owning *view* over a reference-counted buffer, exactly
like NumPy's split between `PyArrayObject` and its data buffer:

- `std::shared_ptr<Buffer> buffer_` — owns the raw allocation + the dtype's
  deleter; multiple arrays/views share one buffer (copy-on-nothing; views alias).
- `DType dtype_`, `std::vector<int64_t> shape_`, `std::vector<int64_t> strides_`
  (byte strides, may be negative or zero), `int64_t offset_` (bytes into buffer),
  `Flags` (C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA, WRITEABLE, ALIGNED).
- Views (`reshape` when compatible, `transpose`, basic slicing, `squeeze`,
  `broadcast_to`) return a new `ndarray` sharing `buffer_` with adjusted
  strides/offset — **no copy**. Operations that cannot be expressed as a stride
  trick (e.g. reshape of a non-contiguous array, fancy indexing) copy.
- Adapt the RAII pattern from matlab_llvm's `MatPtr`/custom deleters and the
  `shape_op<IndexFn>` template (runtime_internal.h) for stride-based shape ops.

Broadcasting follows NumPy rules exactly: align shapes on the trailing
dimension, dims of size 1 stretch (stride set to 0), missing leading dims are
size 1. Broadcasting produces a zero-copy view where possible.

## dtype system

A lightweight value type, not a class hierarchy:

```cpp
enum class DTypeId : uint8_t { Bool, Int8, ..., Float16, Float32, Float64,
                               Complex64, Complex128 };
struct DType { DTypeId id; uint8_t itemsize; /* kind, name helpers */ };
```

- Promotion / `result_type` reproduce NumPy's type-promotion lattice (value-based
  promotion is a non-goal; we implement the NEP 50 dtype-based rules). The
  promotion *table* is ported from NumPy's rules, expressed as pure C++.
- Casting: `can_cast(from, to, casting=no|equiv|safe|same_kind|unsafe)` and a
  strided cast kernel per (from,to) pair. half<->float adapted from NumPy
  `npymath/halffloat`.
- Complex stored interleaved (re,im) to match NumPy's ABI and `.npy` files (note:
  matlab_llvm uses split planes — we diverge here for NumPy compatibility).

## Error model

NumPP reports recoverable usage errors with C++ exceptions, not the CPython
C-API or error codes. A `numpp::error` base (over `std::exception`) roots a typed
hierarchy — `value_error`, `type_error`, `index_error`, `axis_error`,
`linalg_error`, `not_implemented_error` — chosen to mirror the NumPy categories
callers already reason about. Structural/dimensional faults (bad shapes, bad
axes, out-of-range indices, incompatible casts) throw; element-wise numerics
follow IEEE-754 and propagate `NaN`/`inf` instead of throwing. Mutating
operations give the strong exception guarantee: on failure, inputs are unchanged
and no partially-allocated memory leaks. This contract is consumed by every other
capability (each "SHALL raise an error" resolves to one of these types).

## Backend dispatch (the differentiator)

Three-layer model:

1. **Compile-time feature flags** (CMake → `config.hpp`):
   `NUMPP_WITH_BLAS`, `NUMPP_WITH_LAPACK`, `NUMPP_WITH_METAL`, `NUMPP_WITH_VULKAN`,
   `NUMPP_WITH_CUDA`, `NUMPP_WITH_OPENCL`. **All default OFF.** A build with every
   flag off is the mobile baseline and must be fully functional.
2. **Link-time weak backends**: each optional backend is its own static lib
   exposing a small C-ABI vtable (`gemm`, `gesv`, device alloc/copy, kernel
   launch). Absent backends resolve to null — mirrors matlab_llvm's weak-symbol
   `matlab_gpu_gemm` pattern. No backend is a hard dependency of the core.
3. **Runtime selection**: a `CapabilityRegistry` probes at startup (is a BLAS
   vtable linked? does `cuInit`/Metal device enumeration succeed?). Each dispatch
   point picks a backend using `(operation, dtype, problem-size)`:
   - below a per-op size threshold (tunable via env, e.g. `NUMPP_GEMM_MIN`) →
     portable CPU kernel (avoids offload/transpose overhead),
   - GPU only when a device is present *and* data is large enough,
   - **always** a CPU fallback when no accelerated path applies.

`NUMPP_BACKEND` / `NUMPP_GPU_TARGET=cpu|metal|vulkan|cuda|opencl|auto` env vars
force/scope selection (like `MATLAB_GPU_TARGET`); `auto` = Metal on Apple, then
CUDA→Vulkan→OpenCL→CPU elsewhere. Explicitly requesting an uncompiled backend is
a clear error; `auto` silently degrades to CPU.

CPU kernels carry their own internal SIMD tiering (scalar baseline + optional
NEON/AVX via NumPy's portable intrinsics) but that is below this dispatch layer
and never a build requirement.

### GPU backend priority

Metal (iOS/macOS), Vulkan (Android/Linux/Windows), CUDA (NVIDIA), OpenCL
(legacy/cross-vendor) are all in scope as backend *slots*. matlab_llvm already
has CUDA/Metal/OpenCL skeletons (`runtime/gpu/{cuda,metal,opencl}`) to adapt;
Vulkan is new. In this change only the GEMM/matmul dispatch path is wired
end-to-end; broader kernel coverage is Phase 10.

## NumPy oracle harness

Mirrors SymPP's SymPy-oracle approach:

- A test helper shells out to Python (`python3 -c` with NumPy) at test time,
  evaluates the reference expression, and returns the array; the test asserts
  `numpp::allclose(result, oracle, rtol, atol)`.
- **Frozen mode**: a generator script serializes oracle arrays to `.npy` golden
  files checked into `tests/golden/`, so CI can run without Python/NumPy by
  comparing against the frozen data. (Answers the "Yes — NumPy oracle" choice;
  golden freezing keeps CI hermetic.)
- Reading `.npy` in tests is a minimal loader (full `.npy` I/O is Phase 7).

## Build & packaging

- CMake ≥ 3.25, `CMAKE_CXX_STANDARD 20`, clang-first (GCC supported), warnings-
  as-errors, ASan/UBSan options — same option set as SymPP.
- Packaging: `vcpkg.json` + `conanfile.py`. Core has **zero required deps**;
  BLAS/LAPACK/GPU are optional features.
- **iOS/Android contract**: a CI job cross-compiles the core with all backend
  flags OFF using the iOS and Android NDK toolchains and runs the CPU test
  subset, proving the no-accel path stays green. This is the guardrail that keeps
  the portability promise honest.

## Key decisions

- **Interleaved complex** (NumPy ABI) over matlab_llvm's split planes — chosen
  for `.npy` compatibility and oracle parity, at some SIMD cost.
- **Views share buffers via `shared_ptr`** rather than NumPy's manual refcount —
  simpler, RAII, no GIL concerns.
- **Backends as weak C-ABI vtables**, not `#ifdef`-laced kernels — keeps the core
  free of backend headers so mobile builds don't even parse them.
- **dtype-based promotion (NEP 50)** not legacy value-based promotion — matches
  current NumPy and is simpler to specify.

## Risks

- Promotion-rule fidelity is fiddly; the oracle harness is the mitigation —
  generate promotion tables from NumPy and assert parity.
- Weak-symbol/vtable portability across linkers (Apple ld, lld, Android) — verify
  early in the iOS/Android CI job.
- Vulkan compute is the least mature backend; it may slip to a later change
  without affecting the architecture.

## Open questions

- ~~Test framework: Catch2 vs doctest?~~ **Resolved:** the bootstrap uses a tiny
  self-contained harness (`tests/numpp_test.hpp`, ~60 lines) to keep the test
  build zero-dependency and offline/mobile-friendly. Catch2/doctest can be
  swapped in later if richer features are needed.
- Threading: defer a parallel-for/thread-pool decision to the ufunc phase, but
  keep kernel signatures range-based so it can be added without API change.
