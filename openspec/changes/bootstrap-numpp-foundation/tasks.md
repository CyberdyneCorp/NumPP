# Tasks — Bootstrap NumPP foundation

Scope: Phase 0 (foundation, build, oracle, backend skeleton) and Phase 1
(ndarray core + dtype subset). Do NOT implement Phase 2+ features here.

## 0. Project skeleton & build (build-packaging)
- [x] 0.1 Create repo layout: `include/numpp/{core,backend}`, `src/{core,backend}`, `tests/`, `cmake/`, `examples/`
- [x] 0.2 Root `CMakeLists.txt`: CMake 3.25+, C++20, clang-first; options `NUMPP_BUILD_{SHARED,TESTS,EXAMPLES,BENCHMARKS,DOCS}`, `NUMPP_ENABLE_{ASAN,UBSAN}`, `NUMPP_WARNINGS_AS_ERRORS` (mirror SymPP)
- [x] 0.3 Backend feature options (all default OFF): `NUMPP_WITH_{BLAS,LAPACK,METAL,VULKAN,CUDA,OPENCL}`; generate `config.hpp`
- [x] 0.4 `version.hpp.in`, `fwd.hpp`, umbrella `numpp.hpp`
- [x] 0.5 `find_package` export/install rules; `vcpkg.json` + `conanfile.py` (core has zero required deps; backends as features)
- [x] 0.6 CI matrix: clang+gcc on Linux/macOS; ASan/UBSan job; **iOS and Android cross-compile job with all backend flags OFF**
- [x] 0.7 OpenSpec CI gate (`openspec validate --all --strict`) and README skeleton noting the NumPy-oracle + tiered-accel design

## 1. NumPy oracle harness (oracle-testing)
- [x] 1.1 Pick test framework (Catch2 vs doctest — see design open question) and wire into CMake
- [x] 1.2 Minimal `.npy` reader for tests (full I/O is Phase 7)
- [x] 1.3 Live-oracle helper: invoke `python3` + NumPy, return arrays; `allclose(result, oracle, rtol, atol)` with shape+dtype check; exact mode for int/bool
- [x] 1.4 Frozen golden mode: generator writes `tests/golden/*.npy`; comparison mode runs without Python
- [x] 1.5 Skip (not fail) live-oracle tests when Python/NumPy and golden data are both absent
- [x] 1.6 Generate the dtype-promotion oracle table from NumPy for use by dtype tests

## 2. Error model (error-handling)
- [x] 2.0 `numpp::error` base (over `std::exception`) + typed hierarchy: `value_error`, `type_error`, `index_error`, `axis_error`, `linalg_error`, `not_implemented_error`; all carry a message
- [x] 2.0a Establish throw-vs-propagate contract (structural errors throw; element-wise non-finite follows IEEE-754) and strong-exception-safety convention for mutating ops

## 3. dtype system (dtype-system)
- [x] 3.1 `DTypeId` enum + `DType` value type (id, itemsize, kind, name, native byte order) for bool/int*/uint*/float16/32/64/complex64/128
- [x] 3.2 `DType` from name + aliases (`"float64"`, `"f8"`, `"int"`, ...); unknown name -> `type_error`; default platform dtypes (float64/complex128/native int)
- [x] 3.3 half<->float conversion (clean-room IEEE-754 in `core/half.hpp`, round-trip tested; NumPy's SIMD halffloat can be swapped in later)
- [x] 3.4 `result_type` promotion (NEP 50 dtype-based); validate every dtype pair against `numpy.result_type`
- [x] 3.5 `can_cast(from,to,casting)` for no/equiv/safe/same_kind/unsafe; validate against `numpy.can_cast`
- [x] 3.6 Strided cast kernels for all (from,to) pairs incl. real<->complex and half; oracle-checked on non-contiguous inputs

## 4. ndarray core (ndarray-core)
- [x] 4.1 `Buffer` (shared_ptr-owned allocation + deleter) and `ndarray` (buffer, dtype, shape, byte strides, offset, flags incl. C/F/WRITEABLE/ALIGNED). Adapt RAII/deleter pattern from matlab_llvm `MatPtr`
- [x] 4.2 Constructors: from shape+dtype (C-contiguous), from external buffer (no-copy)
- [x] 4.3 Creation routines: `empty/zeros/ones/full` + `*_like`, `eye/identity`, `arange`, `linspace`; default-dtype inference
- [x] 4.4 Contiguity flag computation (C/F) from strides; recompute for views
- [x] 4.5 Stride-based element access + well-defined strided iteration (handles non-contiguous and zero-stride/broadcast views). Adapt `shape_op<IndexFn>` idea from matlab_llvm
- [x] 4.6 Views: `reshape` (view when possible else copy, `-1` inference, error on bad count), `transpose`/`transpose(axes)`/`swapaxes`/`squeeze`, `expand_dims`/newaxis
- [x] 4.7 `ravel` (view when order permits) / `flatten` (always copy)
- [x] 4.8 Basic indexing: integer index (rank-reducing view), slices with negative index/step (negative-stride views); out-of-range -> `index_error`, bad axis -> `axis_error`
- [x] 4.9 `copy(order)`, `ascontiguousarray`/`asfortranarray` (fast path when already satisfied)
- [x] 4.10 Broadcasting: `broadcast_shapes`, `broadcast_to` zero-stride view (read-only); error on incompatible shapes
- [x] 4.11 Writeability/read-only propagation to views; assignment through read-only -> error
- [x] 4.12 Raw data access: typed data pointer, `item(idx...)`, `fill(value)` (writeable-only)
- [x] 4.13 `allclose`/`array_equal` helpers with `equal_nan` + signed-inf semantics (needed by oracle + backend equivalence tests)

## 5. Backend dispatch skeleton (backend-dispatch)
- [x] 5.1 Define backend C-ABI vtable structs (gemm, gesv, device alloc/copy/launch) and nullable registration; core references via pointers (adapt matlab_llvm weak-symbol `matlab_gpu_gemm` pattern)
- [x] 5.2 `CapabilityRegistry`: thread-safe once-init startup probe of linked BLAS/LAPACK + per-GPU device enumeration; query API
- [x] 5.3 Dispatch policy: `(op, dtype, size)` -> backend; per-op size thresholds (env-tunable, e.g. `NUMPP_GEMM_MIN`); guaranteed CPU fallback; introspection of backend actually used
- [x] 5.4 `NUMPP_GPU_TARGET=cpu|metal|vulkan|cuda|opencl|auto` + force-CPU override; `auto` precedence (Metal on Apple else CUDA→Vulkan→OpenCL→CPU); clear error for uncompiled explicit backend
- [x] 5.5 Portable CPU GEMM kernel (always present) as the reference path. Optionally adapt matlab_llvm naive `__restrict__` GEMM; correctness first
- [x] 5.6 Wire ONE accelerated path end-to-end behind `NUMPP_WITH_BLAS`: `cblas_?gemm` (s/d/c/z) with threshold (`blas_backend.cpp`). Compiles conditionally; not exercised in the dev env (no CBLAS dev headers). Metal/Vulkan/CUDA/OpenCL are dispatch slots only.
- [x] 5.7 Equivalence test: CPU-forced vs BLAS-forced matmul agree within tolerance

## 6. Validation & docs
- [x] 6.1 `openspec validate --all --strict` passes
- [x] 6.2 Oracle tests green on desktop (clang + gcc, ASan/UBSan clean). BLAS-off path verified; BLAS-on equivalence test present but skipped locally (no CBLAS dev headers)
- [ ] 6.3 iOS/Android no-accel cross-compile job green
- [x] 6.4 Update README/docs: build matrix, backend flags, oracle usage, error model; note reuse provenance (NumPy / matlab_llvm)
- [ ] 6.5 Archive this change after merge (`openspec archive bootstrap-numpp-foundation`)
