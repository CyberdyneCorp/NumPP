# gpu-kernels Specification

## ADDED Requirements

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
