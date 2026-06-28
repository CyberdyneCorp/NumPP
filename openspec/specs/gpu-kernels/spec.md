# gpu-kernels Specification

## Purpose
TBD - created by archiving change add-gpu-kernels. Update Purpose after archive.
## Requirements
### Requirement: GPU vtable dispatch for ufuncs and reductions
NumPP SHALL route a representative set of element-wise ufuncs (add, subtract,
multiply, divide, negative, sqrt, exp) and reductions (sum, prod) through an
optional weak-linked GpuVTable when one is registered, the operands are
contiguous and float/complex, and the size is at least NUMPP_GPU_MIN; otherwise
it SHALL use the portable CPU kernel. The chosen backend SHALL be reported by
last_backend().

#### Scenario: Device path equals CPU path
- GIVEN a build with a registered device backend
- WHEN an eligible op runs above the size threshold
- THEN the result equals the CPU kernel's result and last_backend() is Device

#### Scenario: Fallback below threshold and for unsupported dtypes
- WHEN an op runs below the threshold, or on an integer dtype
- THEN the CPU kernel is used and last_backend() is Cpu

#### Scenario: Default build unchanged
- GIVEN a build with no device backend (the default)
- WHEN any op runs
- THEN the CPU kernel is always used and all results are unchanged

### Requirement: Real OpenCL GPU backend
NumPP SHALL provide a real OpenCL backend, enabled by NUMPP_WITH_OPENCL, that
registers a GpuVTable executing float32/float64 element-wise arithmetic
(add/sub/mul/div), negate, sqrt and sum/prod reductions on an OpenCL GPU device,
declining exp, complex and integer dtypes to the CPU kernel. It SHALL build
without the system OpenCL headers by linking the ICD loader.

#### Scenario: OpenCL device results equal the CPU path
- GIVEN a build with NUMPP_WITH_OPENCL and an available OpenCL GPU
- WHEN an eligible op runs above the size threshold
- THEN the result equals the CPU kernel's result (IEEE-exact for arithmetic and
  sqrt; within reduction tolerance for sum) and last_backend() is Device

#### Scenario: Graceful fallback when no device is present
- GIVEN NUMPP_WITH_OPENCL is enabled but no OpenCL GPU/ICD is available
- WHEN any op runs
- THEN gpu_vtable() is null and the CPU kernel is used (last_backend() is Cpu)

### Requirement: Real CUDA GPU backend
NumPP SHALL provide a real CUDA backend, enabled by NUMPP_WITH_CUDA, that
registers a GpuVTable executing float32/float64 element-wise arithmetic
(add/sub/mul/div), negate, sqrt and sum/prod reductions with CUDA kernels on an
NVIDIA GPU, declining exp, complex and integer dtypes to the CPU kernel. It SHALL
build to a PTX-virtual architecture so the driver JIT-compiles to the device's
native architecture (allowing an older nvcc to target a newer GPU).

#### Scenario: CUDA device results equal the CPU path
- GIVEN a build with NUMPP_WITH_CUDA and an available NVIDIA GPU
- WHEN an eligible op runs above the size threshold
- THEN the result equals the CPU kernel's result (IEEE-exact for arithmetic and
  sqrt; within reduction tolerance for sum) and last_backend() is Device

#### Scenario: Graceful fallback when no device is present
- GIVEN NUMPP_WITH_CUDA is enabled but no CUDA device is available
- WHEN any op runs
- THEN gpu_vtable() is null and the CPU kernel is used (last_backend() is Cpu)

### Requirement: GPU GEMM dispatch for matmul
The GpuVTable SHALL expose a gemm entry (row-major C[m,n] = A[m,k]·B[k,n],
float32/float64), implemented by the refgpu, OpenCL and CUDA backends. matmul
SHALL route to gpu_vtable()->gemm when a specific GPU backend is targeted (forced
or via NUMPP_GPU_TARGET) and the dtype is float, reporting that backend via
last_backend(); Auto matmul SHALL continue to use BLAS/CPU.

#### Scenario: device matmul matches the CPU result
- GIVEN a build with a GPU backend and an available device
- WHEN matmul is called with that backend forced
- THEN the result matches the CPU gemm within floating-point tolerance and
  last_backend() reports the GPU backend

#### Scenario: gpu_available reflects compiled backend and present device
- WHEN a GPU backend is compiled and a device is present
- THEN CapabilityRegistry::gpu_available returns true for it
- AND requesting an uncompiled/absent GPU backend still raises not_implemented_error

### Requirement: GPU device buffer pooling and tiled GEMM
The CUDA and OpenCL backends SHALL serve device allocations from a bounded reuse
pool (rather than allocating and freeing per call) and SHALL compute GEMM with a
shared-memory tiled kernel, without changing results.

#### Scenario: pooled buffers and tiled GEMM preserve correctness
- WHEN element-wise ops, reductions or GEMM run repeatedly on the device
- THEN results are unchanged (GEMM matches the CPU within floating-point tolerance;
  element-wise arithmetic and sqrt remain IEEE-exact)
- AND device buffers are reused across calls rather than reallocated each time


### Requirement: ScyPP acceleration kernels (sparse / geometry / ndimage)
NumPP SHALL expose, through the same weak-linked GpuVTable and dispatch substrate,
device-accelerable kernels whose high-level API lives in ScyPP (the C++ SciPy
port): CSR sparse matrix-vector product (`csr_spmv`), pairwise squared/euclidean
distance (`cdist_euclidean`), and separable 1-D correlation (`correlate1d`,
matching scipy.ndimage modes reflect/constant/nearest/mirror/wrap). Each SHALL
provide a portable CPU kernel that is always available, auto-select a registered
device backend when the problem clears NUMPP_GPU_MIN, fall back to the CPU kernel
otherwise (or on an unsupported dtype, or when no device is present), and report
the choice via last_backend(). These are NOT part of NumPP's numpy.* surface.

#### Scenario: Device kernel equals the CPU kernel
- GIVEN a build with a registered device backend (real GPU or the reference device)
- WHEN csr_spmv / cdist_euclidean / correlate1d runs above the size threshold
- THEN the result equals the portable CPU kernel's result within tolerance
- AND last_backend() reports the device backend

#### Scenario: CPU fallback below threshold or without a device
- WHEN the problem is below NUMPP_GPU_MIN, the dtype is unsupported, or no device
  backend is registered (the default build)
- THEN the portable CPU kernel computes the result and last_backend() is Cpu

#### Scenario: correlate1d matches scipy.ndimage
- GIVEN weights, an axis, an origin and a boundary mode
- WHEN correlate1d runs
- THEN the result equals scipy.ndimage.correlate1d for the same arguments
