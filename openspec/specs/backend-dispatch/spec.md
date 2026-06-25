# backend-dispatch Specification

## Purpose
TBD - created by archiving change bootstrap-numpp-foundation. Update Purpose after archive.
## Requirements
### Requirement: Portable CPU kernel always available

NumPP SHALL provide a portable, dependency-free C++ implementation of every
numeric operation it exposes. A build with all optional backend feature flags
disabled SHALL compile, link, and pass the CPU test subset on any C++20
toolchain, including iOS and Android cross-compilation toolchains, with no
linear-algebra or GPU dependency.

#### Scenario: No-accel build is fully functional
- GIVEN a build configured with `NUMPP_WITH_BLAS=OFF`, `NUMPP_WITH_LAPACK=OFF`, and all `NUMPP_WITH_<GPU>=OFF`
- WHEN the library is built and the CPU test subset is run
- THEN the build succeeds and all CPU tests pass

#### Scenario: Mobile toolchain build
- GIVEN the iOS and Android NDK cross-compilation toolchains
- WHEN NumPP core is built with all backend flags OFF
- THEN compilation and linking succeed

### Requirement: Compile-time backend feature flags

NumPP SHALL gate every optional backend behind a CMake feature flag surfaced in
a generated `config.hpp`: `NUMPP_WITH_BLAS`, `NUMPP_WITH_LAPACK`,
`NUMPP_WITH_METAL`, `NUMPP_WITH_VULKAN`, `NUMPP_WITH_CUDA`, `NUMPP_WITH_OPENCL`.
All flags SHALL default to OFF. Core translation units SHALL NOT include any
backend SDK header when its flag is OFF.

#### Scenario: Flags default off
- WHEN the project is configured with no backend options specified
- THEN all `NUMPP_WITH_*` backend flags are OFF in the generated config

#### Scenario: Core stays backend-header-free
- GIVEN a build with `NUMPP_WITH_CUDA=OFF`
- WHEN the core sources are compiled
- THEN no CUDA SDK header is included by any core translation unit

### Requirement: Weak-linked backend registration

Each optional backend SHALL be a separate library exposing a small C-ABI vtable
(e.g. `gemm`, `gesv`, device alloc/copy, kernel launch). The core SHALL reference
backends through nullable pointers so that an unlinked backend resolves to absent
rather than an unresolved-symbol link error. No backend SHALL be a hard link
dependency of the core.

#### Scenario: Absent backend resolves to null
- GIVEN a build where the BLAS backend library is not linked
- WHEN the dispatch layer queries the BLAS vtable
- THEN the vtable is reported absent and dispatch proceeds to the CPU kernel

### Requirement: Runtime capability detection

NumPP SHALL probe available backends at startup via a `CapabilityRegistry`:
whether a BLAS/LAPACK vtable is linked, and whether each compiled GPU backend can
enumerate a usable device. The registry SHALL be queryable by the application.

#### Scenario: Report capabilities
- WHEN the capability registry is queried on a CPU-only build
- THEN it reports BLAS, LAPACK, and all GPU backends as unavailable

#### Scenario: GPU present but unusable
- GIVEN a build with a GPU backend compiled in but no usable device at runtime
- WHEN the registry probes that backend
- THEN it reports the backend as unavailable and the CPU path is used

### Requirement: Size-threshold and device dispatch

Each accelerable operation SHALL choose an implementation from
`(operation, dtype, problem size, available backends)`. Below a per-operation
size threshold the portable CPU kernel SHALL be used to avoid offload/setup
overhead; an accelerated backend SHALL be used only when it is available and the
problem is large enough. A CPU fallback SHALL always exist for every operation.

#### Scenario: Small problem uses CPU even when BLAS is present
- GIVEN a build with BLAS available
- WHEN a matmul below the configured `NUMPP_GEMM_MIN` size is executed
- THEN the portable CPU kernel is used

#### Scenario: Large problem uses BLAS when available
- GIVEN a build with BLAS available
- WHEN a matmul above the threshold is executed
- THEN the BLAS backend is used and the result equals the CPU result within tolerance

### Requirement: Backend selection override

NumPP SHALL honor an environment override (`NUMPP_GPU_TARGET` accepting
`cpu|metal|vulkan|cuda|opencl|auto`, and a force-CPU switch) to scope or force
backend selection. `auto` SHALL prefer Metal on Apple platforms and otherwise try
CUDA, then Vulkan, then OpenCL, then CPU. Requesting a backend that was not
compiled in SHALL produce a clear error; `auto` SHALL degrade silently to CPU.

#### Scenario: Force CPU
- GIVEN a build with a GPU backend available
- WHEN `NUMPP_GPU_TARGET=cpu` is set
- THEN all operations use the CPU path

#### Scenario: Auto degrades to CPU
- GIVEN a build with no GPU backend compiled in
- WHEN `NUMPP_GPU_TARGET=auto` is set
- THEN operations run on the CPU path without error

#### Scenario: Explicit unavailable backend errors
- GIVEN a build with `NUMPP_WITH_CUDA=OFF`
- WHEN `NUMPP_GPU_TARGET=cuda` is set
- THEN a clear error is reported

### Requirement: Thread-safe capability registry and dispatch

The capability registry SHALL be initialized exactly once and SHALL be safe to
query concurrently from multiple threads. Dispatch decisions SHALL not mutate
shared state in a way that requires external locking by the caller.

#### Scenario: Concurrent dispatch is safe
- GIVEN the registry has been initialized
- WHEN multiple threads concurrently execute dispatched operations
- THEN no data race occurs and each operation selects a valid backend

### Requirement: Backend selection introspection

NumPP SHALL expose, for testing and diagnostics, which backend served the most
recent dispatched operation (or a per-call out-parameter), so that tests can
assert the CPU-vs-accelerated path actually taken rather than only the numerical
result.

#### Scenario: Observe the chosen backend
- GIVEN a build with BLAS available and a matmul above the threshold
- WHEN the operation runs and the selected backend is queried
- THEN it reports the BLAS backend

#### Scenario: Observe fallback
- GIVEN `NUMPP_GPU_TARGET=cpu`
- WHEN any accelerable operation runs and the selected backend is queried
- THEN it reports the CPU kernel

### Requirement: Result equivalence across backends

For any operation, the result produced by an accelerated backend SHALL equal the
result of the portable CPU kernel within documented floating-point tolerance, so
that enabling a backend never changes observable numerical behavior beyond
rounding.

#### Scenario: BLAS and CPU agree
- GIVEN the same input matrices
- WHEN matmul is run once forced to CPU and once forced to BLAS
- THEN the two results are equal within tolerance (`allclose`)

