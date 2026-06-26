# Complete Tier A — finish partially-done modules (NumPy parity)

## Why
With the Tier 1 + Tier 2 backlog delivered, Tier A of the refreshed
numpy-parity-roadmap closes the remaining gaps in already-existing modules:
array construction from data/buffers, manipulation extras, the rest of the
indexing toolkit (incl. fancy/boolean access), statistics extras, ufunc &
type-check helpers, and the legacy `poly1d` class. These were authored in
parallel (one agent per capability) against fixed headers and oracle-validated.

## What changes
- **array-constructors**: fromiter, frombuffer, broadcast_arrays, meshgrid_sparse, mgrid, ogrid
- **manip-extras**: block, dsplit, trim_zeros, rollaxis
- **indexing-completion**: boolean_index/assign, fancy_index/assign, put_along_axis, place, ix_, fill_diagonal, diag_indices, tril_indices, triu_indices
- **stats-extras**: histogram2d, histogramdd, nanquantile, cov_weighted
- **ufunc-completion**: fix, real_if_close, iscomplex/isreal(+obj), common_type, packbits/unpackbits, emath (sqrt/log/power)
- **poly1d**: poly1d class (call/deriv/integ/roots/+/-/*), polyfit_weighted

## Non-goals
- Fancy indexing as `ndarray::operator[]` subscript sugar (the functional
  equivalents land here); N-D string/structured variants; histogram `density=`
  / `weights=`; emath complex-base power edge cases — deferred to later changes.
