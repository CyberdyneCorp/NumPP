# Extend numpy.ma: arithmetic, comparison masks, per-axis reductions

## Why
The masked-arrays MVP (from tier-c-partial) had only flat reductions and a few
constructors. This adds the deferred arithmetic operators, more masked
constructors, and per-axis reductions — closing most of the numpy.ma gap.

## What changes
- **masked-arrays** capability (numpp::ma):
  - elementwise arithmetic: add, subtract, multiply, divide (result mask =
    a.mask | b.mask)
  - comparison/range mask constructors: masked_less, masked_less_equal,
    masked_greater_equal, masked_not_equal, masked_inside, masked_outside,
    masked_values
  - per-axis reductions: sum_axis, prod_axis, mean_axis, max_axis, min_axis,
    count_axis (an output element is masked iff its whole slice was masked)
  - accessors: getmask, getdata

## Non-goals
- hard/soft masks (harden/soften), masked record/structured arrays, broadcasting
  masked arithmetic beyond same/compatible shapes, and the full numpy.ma method
  surface (anom, mask_rowcols, …) — deferred.
