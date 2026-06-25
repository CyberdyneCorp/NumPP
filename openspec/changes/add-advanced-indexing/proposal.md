# Add advanced indexing routines (NumPy parity, Tier 1)

## Why
Fourth item of the numpy-parity-roadmap backlog. NumPy's gather/scatter and
selection routines (take, take_along_axis, put, diagonal, argwhere, compress,
choose, select, ravel_multi_index, unravel_index) are widely used and were
missing from NumPP.

## What changes
- **advanced-indexing** capability:
  - gather/scatter: take (flat/axis), take_along_axis, put (in-place, flat)
  - extraction: diagonal, argwhere, compress, extract
  - piecewise selection: choose, select
  - flat<->multi index: ravel_multi_index, unravel_index

## Non-goals
- put_along_axis, place, ix_, integer-array/boolean fancy indexing as ndarray
  subscript operators, N-D diagonal append-axis semantics (deferred).
