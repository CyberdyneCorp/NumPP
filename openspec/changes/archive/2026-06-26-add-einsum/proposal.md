# Add einsum & tensor contractions (NumPy parity, Tier 2)

## Why
Eighth item of the numpy-parity-roadmap backlog (first Tier-2 item). Einstein
summation and the tensor-contraction helpers (tensordot, cross, cond, multi_dot)
are high-value and were missing from NumPP.

## What changes
- **einsum** capability:
  - einsum with a general subscript parser (implicit/explicit output, repeated
    indices, summed indices): matmul, trace, diagonal, transpose, sum, outer,
    inner, batched matmul
  - tensordot (axis count or explicit axis lists)
  - cross (last axis length 2 or 3, batched)
  - cond (2-norm condition number via singular values)
  - multi_dot (chained matrix product)

## Non-goals
- einsum optimization/path selection, ellipsis ("...") subscripts, complex-dtype
  einsum, cond for non-2 norms, tensorsolve/tensorinv (deferred).
