# NumPP

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.25%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Modern **C++20 port of [NumPy](https://github.com/numpy/numpy)**. Sibling of
[SymPP](https://github.com/leonardoaraujosantos/SymPP) (the C++ port of SymPy):
clean-room implementation with **NumPy itself wired in as a numerical oracle** —
results are checked against real NumPy via `allclose`.

## What makes NumPP different: tiered acceleration

NumPy assumes a desktop/server with BLAS/LAPACK. NumPP is designed to run
everywhere, including **iOS and Android** where there may be no accelerated
linear algebra and no usable GPU:

- A **portable, dependency-free C++ CPU kernel is always present** and is the
  only thing required to build. A build with every backend flag off is the
  mobile baseline.
- **BLAS/LAPACK** and **GPU** backends (Metal, Vulkan, CUDA, OpenCL) are
  optional, weak-linked, compile-time-gated, and chosen **at runtime** by device
  availability and operation size — always falling back to the CPU kernel.

```
op ──► dispatcher ──► size ≥ threshold & device present? ──► GPU / BLAS backend
                          │ otherwise
                          └──────────────────────────────► portable CPU kernel
```

## Status

Foundation (Phase 0–1) implemented and tested:

- `ndarray`: N-D strided container, shared-buffer views, reshape/transpose/
  squeeze/expand_dims, ravel/flatten, basic & sliced indexing (incl. negative
  steps), broadcasting, read-only views, copy/contiguity, raw data access.
- dtype system: bool / int / uint / float16-64 / complex64-128, NumPy-exact
  `result_type` promotion (NEP 50) and `can_cast`, strided cast kernels.
- Backend dispatch: capability registry, CPU GEMM, optional BLAS GEMM, runtime
  selection with `NUMPP_GPU_TARGET` and size thresholds, backend introspection.
- Error model: `numpp::error` hierarchy mirroring NumPy's categories.
- NumPy oracle test harness (308 checks; the 14×14 promotion table is verified
  against live NumPy).

Roadmap (each a future OpenSpec change): ufuncs & element-wise math, full
linalg, fft, random, `.npy`/`.npz` I/O, sorting/set ops, GPU kernel coverage.
See `openspec/changes/bootstrap-numpp-foundation/proposal.md`.

## Build

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++
cmake --build build
ctest --test-dir build --output-on-failure
```

Default build has **zero required dependencies**. Enable acceleration with
`-DNUMPP_WITH_BLAS=ON`, `-DNUMPP_WITH_CUDA=ON`, etc. The test suite uses Python
+ NumPy as an oracle when available and skips oracle checks otherwise.

## Quick example

```cpp
#include "numpp/numpp.hpp"
using namespace numpp;

ndarray a = arange(0.0, 6.0, 1.0, kFloat64).reshape({2, 3});
ndarray b = a.transpose();          // 3x2 view, no copy
ndarray c = matmul(a, b);           // CPU kernel, or BLAS/GPU if available
```

## Provenance

Adapts proven code rather than reinventing it: the weak-symbol backend tiering
and BLAS dispatch follow the
[matlab_llvm runtime](https://github.com/leonardoaraujosantos/matlab_llvm);
dtype-promotion rules and half-float follow NumPy. Anything coupled to the
CPython C-API is rewritten as pure C++.

## License

MIT — see [LICENSE](LICENSE).
