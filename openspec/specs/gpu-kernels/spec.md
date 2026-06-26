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

