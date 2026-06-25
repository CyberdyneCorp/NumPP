# Add element-wise ufuncs (Phase 3)

## Why

The foundation has `ndarray`, dtypes, broadcasting, and a backend dispatcher, but
no arithmetic. Phase 3 adds NumPy's universal-function (ufunc) layer: element-wise
arithmetic, comparison, logical, bitwise, and math functions, plus reductions —
the layer that makes NumPP broadly useful and that later feeds GPU kernels
(Phase 10).

## What changes

- **ufuncs** capability: a broadcasting element-wise execution engine plus the
  NumPy ufunc set (unary and binary), each with NumPy's type-resolution rules,
  and the core reductions (`sum`, `prod`, `min`, `max`, `mean`, `std`, `var`,
  `any`, `all`) with `axis`/`keepdims`/`dtype`, plus `where`, `clip`, `nonzero`.
- Operator overloads on `ndarray` (`+ - * / ...`) delegating to the ufuncs.

## Reuse vs rewrite

- Reuse the existing broadcasting (`broadcast_shapes`/`broadcast_to`), dtype
  promotion (`result_type`), and cast kernels from the foundation.
- Rewrite ufunc dispatch as pure C++ (NumPy's is `PyObject`-coupled). Math
  kernels use `<cmath>`/`<complex>`.

## Non-goals

- `__array_ufunc__` override protocol, custom user ufuncs, `np.frompyfunc`.
- `out=` broadcasting subtleties beyond exact-shape output (basic `out=` lands;
  fancy cases deferred via issues).
- Object dtype, datetime arithmetic (Phase 9).
