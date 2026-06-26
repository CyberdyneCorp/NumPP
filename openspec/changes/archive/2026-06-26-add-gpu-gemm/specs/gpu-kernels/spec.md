# gpu-kernels Specification

## ADDED Requirements

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
