# Add a real CUDA GPU backend

## Why
The OpenCL backend proved real-GPU execution; this adds a CUDA backend behind the
same weak GpuVTable, validated on an NVIDIA RTX 5060 (Blackwell, sm_120).

## What changes
- **gpu-kernels** capability: a CUDA backend (`backend/cuda_backend.cu`) behind
  `NUMPP_WITH_CUDA`, implementing the GpuVTable for float32/float64 element-wise
  binary (add/sub/mul/div), unary (negate/sqrt) and reductions (sum/prod) with
  CUDA kernels. exp/complex/integer decline to the CPU kernel (arithmetic & sqrt
  are IEEE-exact).
- CMake: `enable_language(CUDA)` when `NUMPP_WITH_CUDA=ON`; the CUDA backend is
  selected ahead of OpenCL/refgpu/null; links `CUDA::cudart`. The target uses a
  **PTX-virtual architecture** (`CUDA_ARCHITECTURES "70-virtual"`) so the driver
  JIT-compiles to the device's native arch — letting an older nvcc (12.0) target
  a newer GPU (sm_120). Host-only warning flags are now language-guarded so they
  are not passed to nvcc.

## Non-goals
- Metal / Vulkan backends; device-resident buffers / async transfer / streams;
  GPU GEMM (matmul still uses BLAS/CPU). The CUDA path copies host buffers per
  call — correctness-first, not performance-tuned. A toolkit matching the GPU's
  native arch (CUDA ≥ 12.8 for sm_120) would allow native SASS instead of PTX JIT.
