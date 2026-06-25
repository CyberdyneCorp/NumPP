# NumPy-parity loop progress

Autonomous loop closing the Tier 1/2 backlog from numpy-parity-roadmap. Each
capability: own OpenSpec change → branch → workflow authors impl+tests →
integrate → oracle-validate vs NumPy → file issue+regression per bug →
PR/merge/archive. Durable tracker (survives context resets).

| # | Capability | Tier | Status |
|--:|-----------|:----:|:------:|
| 1 | array-manipulation (concatenate/stack/split/tile/repeat/flip/roll/pad/...) | 1 | ✅ merged+archived |
| 2 | statistics (cumsum/diff/gradient/median/percentile/cov/corrcoef/digitize + nan*) | 1 | 🟡 next |
| 3 | creation-grids (array/meshgrid/diag/tri*/vander/logspace/geomspace) | 1 | ⬜ |
| 4 | advanced-indexing (take/put/diagonal/argwhere/select/fancy+boolean) | 1 | ⬜ |
| 5 | ufuncs-extras (round/gcd/lcm/sinc/nan_to_num/logaddexp/float_power/modf/...) | 1 | ⬜ |
| 6 | signal-poly (convolve/correlate/interp; polyval/polyfit/roots/poly1d) | 1 | ⬜ |
| 7 | sorting-extras (lexsort/sort_complex) | 1 | ⬜ |
| 8 | einsum (+tensordot/cross/cond/multi_dot) | 2 | ⬜ |
| 9 | random-distributions (full set + Philox/SFC64) | 2 | ⬜ |
| 10 | polynomial-package (Polynomial/Chebyshev/...) | 2 | ⬜ |
| 11 | text-io (savetxt/loadtxt/genfromtxt/print-options) | 2 | ⬜ |
| 12 | char-strings (numpy.char) | 2 | ⬜ |

## Bug log (oracle divergences → issues)
| # | issue | capability | regression test | status |
|---|-------|-----------|-----------------|--------|
| – | (none yet) | | | |
