# NumPP

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.25%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.3.7-brightgreen)](#)
[![Tests](https://img.shields.io/badge/oracle%20checks-2415%20vs%20NumPy-brightgreen)](#)
[![GPU](https://img.shields.io/badge/GPU-OpenCL%20%2B%20CUDA-76B900?logo=nvidia&logoColor=white)](docs/gpu-backends.md)

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
| 6 | **random** | PCG64 / SFC64 / Philox / MT19937 BitGenerators, Generator + legacy RandomState — **bit-exact** with numpy (incl. ziggurat `standard_normal`/`standard_exponential`, `normal`, `choice(replace=False)`); 20+ further distributions (gamma/beta/poisson/binomial/laplace/weibull/multivariate_normal/dirichlet/…), statistically validated |
| 7 | **I/O** | `.npy`/`.npz` save/load (numpy-interop), `loadtxt`/`savetxt`/`genfromtxt`, `fromstring`/`tofile`/`fromfile`, `binary_repr`/`base_repr`, print-options/`array2string`, `array_str`/`array_repr` |
| 8 | **sorting** | sort/argsort/partition, searchsorted(+sorter), unique, argmin/argmax, set ops, bincount, histogram, lexsort, sort_complex |
| 9 | **dtypes++** | fixed-width strings ('U'/'S'), datetime64/timedelta64 (+ business-day `is_busday`/`busday_count`/`busday_offset`), structured/record dtypes, `numpy.char` string ops |
| A | **manipulation & stats** | concatenate/stack/split/tile/repeat/flip/roll/pad/atleast_*/block, cumsum/diff/gradient/percentile/median/cov/corrcoef/digitize + nan*, meshgrid/diag/tri*/vander/logspace, fancy+boolean indexing, take/put/diagonal/choose/select |
| B | **einsum & polynomials** | `einsum` (subscript parser) + tensordot/cross/cond/multi_dot; legacy poly + `numpy.polynomial` (power/Chebyshev/Legendre/Hermite/Laguerre val/vander/roots/der/int + classes); convolve/correlate/interp; ufunc extras, stride tricks (`sliding_window_view`/`as_strided`/`piecewise`) |
| C | **GPU / ma / interop / testing** | **real OpenCL + CUDA backends** (elementwise + reductions + tiled **GEMM**); `numpy.ma` masked arrays (arithmetic, per-axis reductions); `numpy.testing` asserts; **DLPack** + **memmap** |

**2415 oracle checks across 904 cases pass against NumPy 2.1.3** with zero
divergences, on clang and gcc, clean under AddressSanitizer/UBSan.

### Bit-exact randomness
NumPP's `random` module reproduces numpy **bit-for-bit** across all four
BitGenerators (PCG64, SFC64, Philox, MT19937), `Generator` and legacy
`RandomState`: raw streams, `random`/`integers`/`uniform`/`permutation`, the
**ziggurat** `standard_normal`/`standard_exponential` (and hence `normal`), and
`choice(replace=False)`. (Reverse-engineered from numpy's source — e.g. numpy's
Philox uses its own `M0 = 0xD2E7470EE14C6C93` and increments the counter before
generating.)

### Real GPU backends
With `-DNUMPP_WITH_OPENCL=ON` or `-DNUMPP_WITH_CUDA=ON`, float32/float64
arithmetic, `sqrt`, reductions and **matmul** run on the GPU behind the same
weak-vtable dispatch (validated on an NVIDIA RTX 5060; tiled shared-memory GEMM
is ~5–8× the CPU at fp64). Element-wise ops are IEEE-exact vs the CPU path. The
default build stays pure-CPU and dependency-free. See [docs/gpu-backends.md](docs/gpu-backends.md).

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
| `-DNUMPP_WITH_OPENCL=ON` | real OpenCL GPU backend (elementwise/reductions/GEMM) |
| `-DNUMPP_WITH_CUDA=ON` | real CUDA GPU backend (PTX-virtual arch; JITs to newer GPUs) |
| `-DNUMPP_WITH_REFGPU=ON` | CPU-reference device backend (proves the dispatch path) |
| `-DNUMPP_ENABLE_ASAN=ON` | AddressSanitizer |

Packaging via `vcpkg.json` and `conanfile.py`; `find_package(NumPP)` exports the target.

## Examples

`examples/` contains worked programs covering a typical Electrical-Engineering
math course plus a neural network — **each one self-verifies against live NumPy**
at runtime (computes the result in NumPP, recomputes it in NumPy, asserts
`allclose`, prints `PASS`/`FAIL`). All **81 parity assertions across the 11
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
| `ee08_laplace_transfer` | Laplace transfer function — poles/zeros, H(jω) | `linalg::eigvals` (roots), complex Horner |
| `ee09_control_bode` | Control systems — Bode magnitude/phase of a 2nd-order system | `log10`, `angle`, logspace |
| `ee10_windowing` | DSP windows (Hann/Hamming/Blackman) & windowed spectra | `cos`, `fft::fft` |
| `nn01_mlp_xor` | **Neural net** — a 2→4→1 MLP learning XOR with full back-prop | `matmul`, `exp`, broadcasting |

The neural net is implemented end-to-end in NumPP (sigmoid via `1/(1+exp(-x))`,
full-batch gradient descent) and its trained predictions match an identical NumPy
training loop bit-for-bit, converging to XOR (`[0.04, 0.96, 0.96, 0.03]`).

## Developer tasks (`just`)

A [`justfile`](justfile) provides one-word recipes (`just --list`):

```bash
just build         # configure + compile library, tests, examples
just test          # run the test suite (unit + integration vs live NumPy)
just examples      # build + run every example, summarizing NumPy parity
just example nn01_mlp_xor   # run one example
just gcc           # build + test with GCC
just asan          # build + test under ASan/UBSan
just spec          # openspec validate --all --strict
just ci            # full local CI (clang + gcc + asan + spec + examples)
just clean
```

## Scope & deferred items

The full practical NumPy surface is covered and oracle-validated. The remaining
NumPy long-tail is grouped by **why** it's missing; only Buckets A/B stay deferred.
See [docs/numpy-parity-gaps.md](docs/numpy-parity-gaps.md) for the complete roadmap
and per-item rationale.

- **Bucket C — portable, charter-compatible: ✅ shipped.** `errstate`/`seterr`/
  `geterr`, `shares_memory`/`may_share_memory`, `ndindex`/`ndenumerate`/`nditer`,
  array-API linalg aliases (`matrix_transpose`/`vecdot`/`vector_norm`/
  `matrix_norm`/`permute_dims`), `busdaycalendar` (weekmask/holidays), `einsum`
  `optimize=`/`einsum_path`, masked hard/soft masks, polynomial domain/window+`fit`,
  and a variable-length `StringDType`.
- **Bucket A — needs a Python runtime/object model (deferred):** `object` dtype,
  `recarray`, `frompyfunc`/multi-arg `vectorize`, `ctypeslib`, `np.matrix`.
- **Bucket B — needs an external dependency (deferred):** DEFLATE-compressed
  `savez_compressed` (zlib), an FFTW/pocketfft backend, `longdouble`/float128,
  Metal/Vulkan. DLPack and `memmap` interop, and OpenCL + CUDA (tiled GEMM) *are*
  shipped.

Previously-tracked bit-exactness gaps are now **fixed**: ziggurat distributions
(#8), `choice(replace=False)` (#7), standalone `MT19937` (#9) and Philox (#36)
are all bit-exact with numpy.

## License

MIT — see [LICENSE](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for the full
phase-by-phase history and [docs/overview.md](docs/overview.md) for architecture.
