# Add sorting/searching/counting (Phase 8)

## Why
Phase 8 adds numpy's sorting, searching, and counting functions.

## What changes
- **sorting** capability: sort/argsort (along axis, kinds, NaN-last), partition/
  argpartition, searchsorted, unique (+index/inverse/counts), argmin/argmax,
  flatnonzero/count_nonzero, set ops (in1d/isin/intersect1d/union1d/setdiff1d),
  bincount, histogram/histogram_bin_edges.

## Non-goals
- Sorting structured/string dtypes (Phase 9), lexsort, sort with order=.
