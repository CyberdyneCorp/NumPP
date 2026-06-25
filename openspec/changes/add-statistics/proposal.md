# Add statistics & cumulative routines (NumPy parity, Tier 1)

## Why
Second item of the numpy-parity-roadmap backlog. NumPy's cumulative, difference,
and order-statistics routines (cumsum, diff, gradient, median, percentile, cov,
corrcoef, …) are everyday tools and were missing from NumPP.

## What changes
- **statistics** capability:
  - cumulative: cumsum/cumprod/nancumsum/nancumprod (axis or flattened)
  - differences: diff (n,axis)/ediff1d/gradient (1-D)/ptp
  - order statistics (linear interpolation): median/percentile/quantile +
    nanmedian/nanpercentile
  - weighted/correlation/binning: average (weighted)/cov/corrcoef/digitize
  - nan-aware index reductions: nanargmin/nanargmax

## Non-goals
- histogram2d/histogramdd, nanquantile alias, other interpolation methods,
  weighted cov (`fweights`/`aweights`), 2-D gradient with spacing — deferred.
