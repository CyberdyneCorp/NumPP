# Tasks — OpenCL GPU backend

- [x] vendored minimal OpenCL declarations (no system CL headers needed)
- [x] OpenCL backend: float32/float64 elementwise binary/unary + reductions on the device
- [x] decline exp / complex / integer dtypes to the CPU kernel (correctness)
- [x] CMake wiring: NUMPP_WITH_OPENCL ahead of REFGPU/null; link the ICD loader
- [x] validated on a real device (NVIDIA RTX 5060): test_gpu device path ACTIVE,
      results equal CPU; full suite 625/625 with OpenCL ON
- [x] default build (no OpenCL) unchanged; openspec validate; PR + merge + archive
