# Add sorting extras (NumPy parity, Tier 1)

## Why
Seventh item of the numpy-parity-roadmap backlog. NumPy's indirect multi-key
sort (lexsort), complex sort (sort_complex) and sorter-based searchsorted were
missing from NumPP.

## What changes
- **sorting-extras** capability:
  - lexsort (stable multi-key indirect sort; last key primary)
  - sort_complex (sort by real then imaginary part)
  - searchsorted overload accepting a `sorter` index array

## Non-goals
- array-valued kth for partition, msort (deprecated in numpy), structured-key
  lexsort beyond numeric keys (deferred).
