# Tasks — GPU GEMM

- [x] add gemm slot to GpuVTable
- [x] implement gemm (float32/float64, p-order) in refgpu, OpenCL and CUDA backends
- [x] matmul dispatches to GPU gemm for explicitly-targeted GPU backends; last_backend reports it
- [x] gpu_available reflects compiled backend + present device
- [x] GPU gemm test (tries each backend, skips if not compiled); matches CPU within 1e-10
- [x] fix: tests include config.hpp so NUMPP_WITH_* guards work in feature builds
- [x] validated on RTX 5060: CUDA + OpenCL gemm on device; all 3 configs 626/626; ASan/UBSan green
- [x] openspec validate; PR + merge + archive
