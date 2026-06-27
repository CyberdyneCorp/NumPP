# NumPP architecture overview

## Layout

```
include/numpp/
  numpp.hpp            umbrella header
  core/                ndarray, dtype (+ extended-dtype metadata), shape, error, half, creation
  umath/               ufuncs + reductions (the element-wise engine)
  mathx/               extra math ufuncs (around/gcd/sinc/logaddexp/modf/...), type-check, emath
  manip/  grids/  construct/  indexing/   manipulation, coordinate grids, constructors, fancy/boolean indexing
  stats/               cumulative/order statistics, histogram2d/dd, weighted cov
  linalg/  tensor/      solvers & decompositions; einsum, tensordot/cross/cond/multi_dot
  poly/  polynomial/    legacy poly* + poly1d; numpy.polynomial (orthogonal bases + classes)
  fft/                 DFT engine + transforms
  random/              SeedSequence, PCG64/SFC64/Philox/MT19937, Generator, RandomState,
                       ziggurat tables, distributions (bit-exact + statistical)
  io/  textio/          npy/npz, printing/print-options; loadtxt/savetxt/fromfile, *_repr
  sorting/             sort/search/unique/set-ops/histogram, lexsort/sort_complex
  strings/ datetime/ struct/   extended dtypes: 'U'/'S', numpy.char, datetime + busday, records
  ma/  testing/  lib/  interop/   masked arrays, numpy.testing, stride tricks, DLPack + memmap
  backend/             dispatch, BLAS/LAPACK/GPU weak vtables (OpenCL/CUDA/refgpu), config
src/<module>/          implementations mirroring include/
tests/                 self-contained harness + NumPy oracle (golden-frozen mode)
```

## Core model

`ndarray` is a non-owning view over a reference-counted `Buffer`: dtype, shape,
byte strides, offset, flags. Views (slicing, transpose, broadcast, reshape-when-
contiguous) share the buffer; only operations that can't be expressed as a stride
trick copy. A generic strided iterator drives every kernel, so non-contiguous and
broadcasted arrays work without materializing.

`DType` is a value type: the 14 numeric dtypes are a plain enum id; extended
dtypes (string/bytes/datetime/timedelta/struct) carry a shared
`DTypeMeta` side-channel (itemsize, unit, field table) that is **null for numeric
dtypes**, so they remain cheap and unaffected.

## Backend dispatch

A compile-time flag gates each optional backend; a link-time **weak vtable**
(nullable function-pointer table) exposes it; a runtime check selects it by
`(operation, dtype, size)`, always falling back to the portable CPU kernel. The
same model serves BLAS (GEMM), LAPACK (solve), and the GPU device path —
element-wise ufuncs, reductions and **GEMM/matmul**. Two real GPU backends
implement that path: **OpenCL** (`-DNUMPP_WITH_OPENCL=ON`) and **CUDA**
(`-DNUMPP_WITH_CUDA=ON`), both with a device-buffer reuse pool and a tiled
shared-memory GEMM; element-wise arithmetic/sqrt are IEEE-exact vs the CPU. A
CPU-reference device (`NUMPP_WITH_REFGPU`) proves the dispatch on hosts with no
GPU. `last_backend()` reports what actually ran. This is what keeps the library
buildable and correct on hardware with no accelerators (iOS/Android included),
while letting it use a GPU when one is present. See `docs/gpu-backends.md`.

## Validation

Every feature is checked against real NumPy: a test helper evaluates the NumPy
expression and asserts `allclose` (or exact equality for integers/bitstreams).
A frozen-golden mode lets CI run without Python. Sign/order-ambiguous results
(qr/svd/eig) are validated by reconstruction (`A=QR`, `A=UΣVᴴ`, `Av=λv`).
The four BitGenerators and the ziggurat/`choice` paths are checked **bit-exact**
against numpy's raw streams; distributions whose numpy algorithm isn't replicated
in portable C++ are validated **statistically** (large-sample moments vs the
closed-form theory). 2083 oracle checks across 770 cases pass with zero
divergences.
