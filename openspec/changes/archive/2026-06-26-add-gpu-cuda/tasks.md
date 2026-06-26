# Tasks — CUDA GPU backend

- [x] CUDA backend (backend/cuda_backend.cu): float32/float64 elementwise binary/unary + reductions
- [x] decline exp / complex / integer dtypes to the CPU kernel (IEEE-exact device results)
- [x] CMake: enable_language(CUDA) under NUMPP_WITH_CUDA; select ahead of OpenCL/refgpu/null; link CUDA::cudart
- [x] PTX-virtual arch (70-virtual) so the driver JITs to sm_120 (older nvcc -> newer GPU)
- [x] language-guard host warning flags so nvcc does not receive -Wall/-Werror
- [x] validated on NVIDIA RTX 5060: test_gpu device path ACTIVE, full suite 625/625; default build unchanged
- [x] openspec validate; PR + merge + archive
