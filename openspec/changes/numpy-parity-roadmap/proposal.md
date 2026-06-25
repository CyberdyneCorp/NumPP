# NumPy parity roadmap (gap backlog)

## Why
NumPP v1.0.0 covers the core NumPy surface (ndarray, dtypes incl. strings/
datetime/structured, ufuncs, reductions, sorting, linalg, fft, random, I/O).
This change records — as a living backlog — the NumPy features that are still
missing, prioritized into tiers, so future work flows through OpenSpec.

The full per-module gap analysis lives in `docs/numpy-parity-gaps.md`. This
change is intentionally **not implemented/archived**; it is the tracking artifact.
Each tier item below should graduate into its own OpenSpec change when picked up.

## What changes
- **numpy-parity** capability: a documented set of target requirements for the
  missing NumPy areas (array manipulation, statistics, grids/indexing, ufunc
  extras, polynomials, einsum, full random distributions, text I/O, char ops),
  plus the suggested implementation order.

## Tiers (summary)
- **Tier 1** (do first): array manipulation (concatenate/stack/split/tile/repeat/
  flip/roll/pad/atleast_*), cumulative+diff (cumsum/cumprod/diff/gradient/ptp),
  statistics (median/percentile/quantile/average/cov/corrcoef/digitize + nan*),
  creation/grids (meshgrid/diag/tri*/vander/logspace), advanced indexing
  (take/put/diagonal/argwhere/fancy+boolean), ufunc extras (round/gcd/lcm/sinc/
  nan_to_num/logaddexp/float_power/modf/frexp/ldexp/divmod), signal/poly basics
  (convolve/correlate/interp, polyval/polyfit/roots/poly1d), sorting extras
  (lexsort/sort_complex).
- **Tier 2**: einsum (+tensordot/cross/cond/multi_dot), full random distribution
  set (+Philox/SFC64), numpy.polynomial package, text/binary I/O
  (savetxt/loadtxt/genfromtxt/print-options/scientific), numpy.char strings.
- **Tier 3**: masked arrays, object dtype/recarray/testing, datetime business-day
  functions, real GPU backends, and the tracked bit-exact long-tail (#7/#8/#9/#11/#14).

## Non-goals
- Implementing any tier here; this change is the roadmap. Implementation lands as
  separate changes (add-array-manipulation, add-statistics, add-einsum, ...).
