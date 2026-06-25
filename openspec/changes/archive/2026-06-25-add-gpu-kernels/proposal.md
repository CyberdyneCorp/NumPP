# Add GPU kernel dispatch (Phase 10)

## Why
The backend layer dispatched only GEMM to accelerated backends. Phase 10 extends
the weak-vtable model to element-wise ufuncs and reductions on a 'device', with
the portable CPU kernel always the fallback — proving the path end-to-end via a
CPU-reference device (no real GPU required in this environment).

## What changes
- **gpu-kernels** capability: a GpuVTable (elementwise binary/unary + reduction,
  by op code + dtype), a nullable gpu_vtable() (null by default), and a
  CPU-reference backend behind NUMPP_WITH_REFGPU. The ufunc layer routes
  add/subtract/multiply/divide, negative/sqrt/exp, and sum/prod through the
  device when present, contiguous, float/complex, and above NUMPP_GPU_MIN;
  Backend::Device + last_backend() introspection.

## Non-goals
- Real Metal/CUDA/Vulkan/OpenCL kernels (the vtable slots exist; implementations
  are future work). Device memory residency / async transfer.
