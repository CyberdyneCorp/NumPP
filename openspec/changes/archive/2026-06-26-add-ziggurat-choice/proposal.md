# Bit-exact ziggurat (#8) and choice(replace=False) (#7)

## Why
The last two bit-exact long-tail items. Both cracked by replicating numpy's exact
algorithm (and, for the ziggurat, its exact lookup tables) from numpy's source.

## What changes
- **random** capability:
  - **standard_normal / standard_exponential** are now bit-exact with numpy
    (ziggurat). Implements numpy's `random_standard_normal` / `random_standard_exponential`
    consuming the PCG64 stream identically, using numpy's 256-entry tables
    (`ki/wi/fi_double`, `ke/we/fe_double`) and constants, extracted verbatim into
    `src/random/ziggurat_constants.h`. This also makes **normal** bit-exact
    (`loc + scale*standard_normal`). (#8)
  - **choice(replace=False)** is now bit-exact: numpy's Floyd open-addressed
    hash-set algorithm for the common case and the `_shuffle_int` tail-shuffle
    branch for large/dense populations (cutoff 50), using the Lemire bounded
    integer path. (#7)

## Non-goals
- gamma/beta/poisson/binomial and the derived distributions remain statistically
  correct but not bit-exact (their algorithms differ from numpy's); shuffle /
  permutation still use masked rejection (not the Lemire `_shuffle_int`).
