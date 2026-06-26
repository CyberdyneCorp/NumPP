# Add GPU GEMM (matmul on the device)

## Why
The GPU backends covered element-wise ops and reductions but not matrix multiply.
This adds a GEMM slot to the GpuVTable and routes `matmul` to the device when a
GPU backend is explicitly targeted — validated on an NVIDIA RTX 5060 via both
CUDA and OpenCL.

## What changes
- **gpu-kernels** capability:
  - `GpuVTable` gains a `gemm(dt, m, n, k, A, B, C)` slot (row-major
    C = A·B, p-order accumulation). Implemented in the refgpu, OpenCL and CUDA
    backends for float32/float64.
  - `matmul` dispatches to `gpu_vtable()->gemm` when a specific GPU backend is
    targeted (forced or `NUMPP_GPU_TARGET=cuda|opencl`) and the dtype is float;
    `Auto` still uses BLAS/CPU. `last_backend()` reports the GPU backend.
  - `CapabilityRegistry::gpu_available` now returns true when a backend is
    compiled and a device registered a vtable (was a hardwired placeholder).
- Fixes a latent test bug: `config.hpp` was not included by the tests, so the
  `NUMPP_WITH_*` `#if` guards were inert; now included and the GPU feature builds
  exercise the guarded paths correctly.

## Non-goals
- GEMM is **not** bitwise-identical to the CPU path: device FMA contraction makes
  it accurate (matches within ~1e-10) but not bit-exact. Element-wise add/sqrt
  remain IEEE-exact.
- `Auto`-routing matmul to the GPU (kept on BLAS/CPU to preserve default
  behaviour); tiled/shared-memory GEMM, complex dtypes, device-resident buffers.
