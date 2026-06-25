# sorting Specification

## ADDED Requirements

### Requirement: Sort and argsort
NumPP SHALL provide `sort` and `argsort` along an axis (default last) and over a
flattened array (axis=None), with quicksort/mergesort(stable)/heapsort kinds,
ascending order, NaN sorted last, matching numpy.

#### Scenario: Sort matches numpy
- WHEN `sort(a)` is computed
- THEN it equals `numpy.sort(a)` including NaN placed at the end

#### Scenario: argsort reconstructs the sort
- WHEN `argsort(a)` returns indices
- THEN indexing `a` by them yields `sort(a)`; the stable kind preserves tie order

### Requirement: Searching and selection
NumPP SHALL provide `searchsorted` (left/right), `partition`/`argpartition`
(kth), `argmin`/`argmax` (axis), `flatnonzero`, and `count_nonzero`, matching numpy.

#### Scenario: searchsorted
- WHEN `searchsorted(a, v, side)` is computed on a sorted `a`
- THEN it equals `numpy.searchsorted`

#### Scenario: partition kth
- WHEN `partition(a, k)` is computed
- THEN the kth element is in its final sorted position and the partition property holds

### Requirement: Unique, set ops, and counts
NumPP SHALL provide `unique` (+return_index/inverse/counts), the set operations
`in1d`/`isin`/`intersect1d`/`union1d`/`setdiff1d`, `bincount`, and `histogram`/
`histogram_bin_edges`, matching numpy.

#### Scenario: unique with counts
- WHEN `unique(a, return_counts=True)` is computed
- THEN the values and counts equal numpy's

#### Scenario: histogram
- WHEN `histogram(a, bins)` is computed
- THEN the counts and bin edges equal `numpy.histogram`
