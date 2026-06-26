# Tasks — GPU performance

- [x] device buffer reuse pool (size-bucketed, capped) in CUDA and OpenCL backends
- [x] route elementwise / reduce / gemm allocations through the pool
- [x] 16x16 shared-memory tiled GEMM kernel (CUDA __shared__, OpenCL __local)
- [x] correctness unchanged: CUDA + OpenCL 626/626; GEMM within 1e-10
- [x] benchmark: 5–8x speedup vs CPU on fp64 matmul (109 GFLOP/s @ 1024^3, RTX 5060)
- [x] default build unchanged; openspec validate; PR + merge + archive
