# Tasks — Phase 10 GPU kernels

- [x] 1 GpuVTable interface + nullable gpu_vtable() + gpu_null.cpp
- [x] 2 CPU-reference device backend behind NUMPP_WITH_REFGPU; CMake wiring
- [x] 3 Route add/sub/mul/div, negative/sqrt/exp, sum/prod through device + Backend::Device
- [x] 4 Tests: default unchanged + device==CPU and last_backend correct; both configs clang+gcc+ASan
- [x] 5 openspec validate --strict; PR + merge + archive
