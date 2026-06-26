# gpu-kernels Specification

## ADDED Requirements

### Requirement: GPU device buffer pooling and tiled GEMM
The CUDA and OpenCL backends SHALL serve device allocations from a bounded reuse
pool (rather than allocating and freeing per call) and SHALL compute GEMM with a
shared-memory tiled kernel, without changing results.

#### Scenario: pooled buffers and tiled GEMM preserve correctness
- WHEN element-wise ops, reductions or GEMM run repeatedly on the device
- THEN results are unchanged (GEMM matches the CPU within floating-point tolerance;
  element-wise arithmetic and sqrt remain IEEE-exact)
- AND device buffers are reused across calls rather than reallocated each time
