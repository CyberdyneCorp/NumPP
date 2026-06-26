# GPU performance: device buffer pool + tiled GEMM

## Why
The OpenCL and CUDA backends allocated/freed device memory on every call and used
a naive one-thread-per-element GEMM. This adds a device-buffer reuse pool and a
shared-memory tiled GEMM, giving a measured 5–8x speedup over the CPU on fp64
matmul (109 GFLOP/s at 1024³ on an RTX 5060) without changing results.

## What changes
- **gpu-kernels** capability (CUDA and OpenCL backends):
  - **Device buffer pool**: device allocations (cudaMalloc / clCreateBuffer) are
    served from a small size-bucketed reuse pool (capped at 32 buffers) instead of
    allocating and freeing per call — eliminating per-op allocation overhead.
  - **Tiled GEMM**: a 16×16 shared-memory (`__shared__` / `__local`) tiled kernel
    replaces the naive GEMM in both backends.

## Non-goals
- Results are unchanged (GEMM still matches the CPU within ~1e-10; element-wise
  add/sqrt still IEEE-exact). No API change.
- True device-resident ndarrays (data staying on the GPU across operations) — the
  pool reuses device *allocations*; host↔device copies still happen per call.
  Pinned/unified memory, multi-stream overlap and cuBLAS/clBLAST GEMM are future
  work.
