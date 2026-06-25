# NumPP

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.25%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen)](#)
[![Tests](https://img.shields.io/badge/oracle%20checks-960%20vs%20NumPy-brightgreen)](#)

Modern **C++20 port of [NumPy](https://github.com/numpy/numpy)** — a clean-room
implementation validated against real NumPy as a numerical oracle (sibling of
[SymPP](https://github.com/leonardoaraujosantos/SymPP), the SymPy port).

```cpp
#include "numpp/numpp.hpp"
using namespace numpp;

ndarray a = arange(0.0, 6.0, 1.0, kFloat64).reshape({2, 3});
ndarray b = a.transpose();                  // zero-copy view
ndarray c = matmul(a, b);                   // CPU, or BLAS/GPU if linked
auto [w, V] = std::tie(linalg::eigh(c).eigenvalues, linalg::eigh(c).eigenvectors);
auto spectrum = fft::fft(a.reshape({6}));   // FFT
auto rng = random::default_rng(42);         // bit-exact with numpy.random.PCG64
```

## What makes NumPP different: tiered acceleration

NumPy assumes a desktop/server with BLAS/LAPACK. NumPP is designed to run
**everywhere, including iOS and Android** where there may be no accelerated
linear algebra and no usable GPU:

- A **portable, dependency-free C++ CPU kernel is always present** and is the only
  thing required to build (`ldd` shows zero extra deps).
- **BLAS/LAPACK** and **GPU/device** backends are optional, weak-linked,
  compile-time-gated, and selected **at runtime** by availability and operation
  size — always falling back to the CPU kernel.

```
op ──► dispatcher ──► device present & size ≥ threshold? ──► BLAS / device backend
                          │ otherwise
                          └────────────────────────────────► portable CPU kernel
```

## Parity status (oracle-validated against NumPy)

| # | Module | Coverage |
|--:|--------|----------|
| 0–2 | **Core** | ndarray (N-D strided views, reshape/transpose/broadcast/indexing), dtypes (bool/int/uint/float16-64/complex), NEP-50 promotion, casting, error model |
| 3 | **ufuncs** | arithmetic, comparison, logical, bitwise, shifts, full unary math + trig/hyperbolic, predicates, reductions (sum/prod/min/max/mean/var/std/any/all), where, clip, nonzero, operators with weak-promoted scalars, out=/where= |
| 4 | **linalg** | dot/vdot/inner/outer/trace/kron, solve/inv/det/slogdet/matrix_power, cholesky, qr, eigh/eigvalsh, svd/svdvals/pinv/matrix_rank/lstsq, eig/eigvals, norms (all orders) |
| 5 | **fft** | fft/ifft (Cooley-Tukey + Bluestein for any n), rfft/irfft, hfft/ihfft, fft2/fftn + real variants, fftfreq/fftshift |
| 6 | **random** | PCG64 + MT19937 BitGenerators, Generator + legacy RandomState — **bit-exact** seeded streams; distributions (normal/exponential/gamma/beta/chisquare/poisson/binomial) |
| 7 | **I/O** | `.npy`/`.npz` save/load (numpy-interop both ways), numpy-compatible `array_str`/`array_repr` |
| 8 | **sorting** | sort/argsort (kinds, axis, NaN-last), partition/argpartition, searchsorted, unique, argmin/argmax, set ops, bincount, histogram |
| 9 | **dtypes++** | fixed-width strings ('U'/'S'), datetime64/timedelta64, structured/record dtypes (field views) |
| 10 | **device** | weak-vtable GPU dispatch for ufuncs/reductions (CPU-reference device proves the path) |

**960 oracle checks pass against NumPy 2.1.3** with zero divergences, on clang
and gcc, clean under AddressSanitizer/UBSan.

### Bit-exact randomness
A seeded `numpp::random::Generator(seed)` reproduces
`numpy.random.Generator(np.random.PCG64(seed))` **bit-for-bit** (raw stream,
`random`, `integers`, `uniform`, `permutation`); `RandomState(seed)` matches the
legacy MT19937 stream exactly.

## Build

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++
cmake --build build
ctest --test-dir build --output-on-failure
```

Default build has **zero required dependencies**. Optional backends:

| Flag | Effect |
|------|--------|
| `-DNUMPP_WITH_BLAS=ON` | route GEMM through CBLAS |
| `-DNUMPP_WITH_LAPACK=ON` | route solve etc. through LAPACKE |
| `-DNUMPP_WITH_REFGPU=ON` | enable the CPU-reference device backend |
| `-DNUMPP_ENABLE_ASAN=ON` | AddressSanitizer |

Packaging via `vcpkg.json` and `conanfile.py`; `find_package(NumPP)` exports the target.

## Examples

`examples/` contains worked programs covering a typical Electrical-Engineering
math course plus a neural network — **each one self-verifies against live NumPy**
at runtime (computes the result in NumPP, recomputes it in NumPy, asserts
`allclose`, prints `PASS`/`FAIL`). All **64 parity assertions across the 8
examples pass** against NumPy 2.1.3.

| Example | EE / ML topic | NumPP features |
|---------|---------------|----------------|
| `ee01_rlc_impedance` | Series RLC AC analysis — complex impedance, phase, resonance | complex ufuncs, `arctan2`, `absolute` |
| `ee02_nodal_analysis` | Resistive circuit nodal analysis — `G·v = I` | `linalg::solve`, `det`, `dot` |
| `ee03_fourier_spectrum` | Fourier analysis of a sampled signal, peak detection | `fft::fft`, `fftfreq` |
| `ee04_convolution_filter` | FIR filtering via the convolution theorem | `fft`/`ifft`, `real` |
| `ee05_state_space_stability` | State-space eigen-stability | `linalg::eigvals`, `trace`, `det` |
| `ee06_least_squares_fit` | Least-squares polynomial regression | `linalg::lstsq`, Vandermonde |
| `ee07_three_phase_power` | Three-phase AC power — phasors, P/Q, power factor | complex arithmetic, `conj` |
| `nn01_mlp_xor` | **Neural net** — a 2→4→1 MLP learning XOR with full back-prop | `matmul`, `exp`, broadcasting |

The neural net is implemented end-to-end in NumPP (sigmoid via `1/(1+exp(-x))`,
full-batch gradient descent) and its trained predictions match an identical NumPy
training loop bit-for-bit, converging to XOR (`[0.04, 0.96, 0.96, 0.03]`).

```bash
cmake -S . -B build -DNUMPP_BUILD_EXAMPLES=ON && cmake --build build
./build/examples/numpp_nn01_mlp_xor        # or any numpp_ee0*
```

## Known limitations (tracked as GitHub issues)

These are *correct-but-not-bit/format-exact* or deferred — not correctness bugs:

- Distributions use standard samplers (not numpy's ziggurat), validated
  statistically (#8); `choice(replace=False)` is a valid sampler with a different
  stream (#7); standalone `MT19937` raw stream differs (RandomState is exact) (#9).
- Array printing is fixed-notation only — no scientific switch yet (#11).
- Structured-dtype `.npy` save/load not yet implemented (#14).
- Some ufunc long-tail (nan-reductions shipped; argmin/cumsum pending) (#3).

## License

MIT — see [LICENSE](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for the full
phase-by-phase history and [docs/overview.md](docs/overview.md) for architecture.
