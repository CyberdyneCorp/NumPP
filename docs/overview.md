# NumPP architecture overview

## Layout

```
include/numpp/
  numpp.hpp            umbrella header
  core/                ndarray, dtype (+ extended-dtype metadata), shape, error, half
  umath/               ufuncs + reductions (the element-wise engine)
  linalg/              solvers and decompositions
  fft/                 DFT engine + transforms
  random/              SeedSequence, PCG64, MT19937, Generator, RandomState
  io/                  npy/npz, array printing
  sorting/             sort/search/unique/set-ops/histogram
  strings/ datetime/ struct/   extended dtype support
  backend/             dispatch, BLAS/LAPACK/GPU weak vtables, config
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
same model serves BLAS (GEMM), LAPACK (solve), and the device path (element-wise
ufuncs + reductions). `last_backend()` reports what actually ran. This is what
keeps the library buildable and correct on hardware with no accelerators.

## Validation

Every feature is checked against real NumPy: a test helper evaluates the NumPy
expression and asserts `allclose` (or exact equality for integers/bitstreams).
A frozen-golden mode lets CI run without Python. Sign/order-ambiguous results
(qr/svd/eig) are validated by reconstruction (`A=QR`, `A=UΣVᴴ`, `Av=λv`).
