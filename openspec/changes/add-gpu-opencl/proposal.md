# Add a real OpenCL GPU backend

## Why
The GPU dispatch architecture (weak GpuVTable + size threshold + last_backend)
was proven only by the CPU-reference device. This adds a **real** GPU backend
that executes kernels on an actual OpenCL device (validated on an NVIDIA
RTX 5060), closing the achievable part of the add-gpu-backends roadmap item.

## What changes
- **gpu-kernels** capability: a real OpenCL backend (`backend/opencl_backend.cpp`)
  behind the existing `NUMPP_WITH_OPENCL` flag, implementing the GpuVTable for
  float32/float64 element-wise binary (add/sub/mul/div), unary (negate/sqrt) and
  reductions (sum/prod) on the GPU. exp and complex/integer dtypes decline to the
  CPU kernel (exp is ULP-approximate on device; arithmetic/sqrt are IEEE-exact).
- Vendored minimal OpenCL declarations (`backend/opencl_min.h`) so it builds
  without the system CL headers, linking the ICD loader (libOpenCL.so.1).
- CMake wires `NUMPP_WITH_OPENCL` (real OpenCL) ahead of `NUMPP_WITH_REFGPU`
  (CPU-reference) and the null backend.

## Non-goals
- **CUDA backend**: requires the CUDA toolkit (nvcc + cudart); not installed in
  this environment. The same vtable shape applies once a CUDA TU can be compiled.
- Metal / Vulkan backends; device-resident buffers / async transfer / pinned
  memory; GPU GEMM (matmul still uses BLAS/CPU). The OpenCL path copies host
  buffers per call — correctness-first, not a performance-tuned backend.
