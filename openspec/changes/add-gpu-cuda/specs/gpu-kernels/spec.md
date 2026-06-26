# gpu-kernels Specification

## ADDED Requirements

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
