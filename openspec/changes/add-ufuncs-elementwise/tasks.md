# Tasks — Phase 3 ufuncs

- [ ] 1 Element-wise engine: broadcasted unary/binary loops over compute dtype; type-resolution policies (arithmetic, true-divide, transcendental→float, comparison→bool, bitwise/logical)
- [ ] 2 Arithmetic ufuncs: add, subtract, multiply, true_divide, floor_divide, mod, power, negative, positive, absolute, sign
- [ ] 3 Comparison ufuncs: equal, not_equal, less, less_equal, greater, greater_equal (bool)
- [ ] 4 Logical: logical_and/or/xor/not; Bitwise: bitwise_and/or/xor/invert, left_shift, right_shift
- [ ] 5 Math: sqrt, cbrt, square, reciprocal, exp, expm1, log, log2, log10, log1p
- [ ] 6 Trig/hyperbolic: sin, cos, tan, arcsin, arccos, arctan, arctan2, sinh, cosh, tanh, arcsinh, arccosh, arctanh, deg2rad, rad2deg
- [ ] 7 Float ops: floor, ceil, trunc, rint, copysign, hypot, fabs, signbit, isnan, isinf, isfinite; minimum/maximum/fmin/fmax; clip
- [ ] 8 Reductions: sum, prod, min, max, mean, std, var, any, all (axis, keepdims, dtype)
- [ ] 9 where, nonzero
- [ ] 10 Operator overloads on ndarray delegating to ufuncs
- [ ] 11 kwargs parity: out=, where=, keepdims=, axis tuples, dtype=
- [ ] 12 NumPy-oracle unit + integration tests for every ufunc and reduction; regression tests + GitHub issues for any divergence
- [ ] 13 openspec validate --strict; build clang+gcc, ASan/UBSan clean; archive
