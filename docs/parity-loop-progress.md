# NumPy-parity loop progress

Autonomous loop closing the Tier 1/2 backlog from numpy-parity-roadmap. Each
capability: own OpenSpec change → branch → workflow authors impl+tests →
integrate → oracle-validate vs NumPy → file issue+regression per bug →
PR/merge/archive. Durable tracker (survives context resets).

| # | Capability | Tier | Status |
|--:|-----------|:----:|:------:|
| 1 | array-manipulation (concatenate/stack/split/tile/repeat/flip/roll/pad/...) | 1 | ✅ merged+archived |
| 2 | statistics (cumsum/diff/gradient/median/percentile/cov/corrcoef/digitize + nan*) | 1 | ✅ merged+archived |
| 3 | creation-grids (meshgrid/indices/diag/diagflat/tri*/vander/logspace/geomspace/fromfunction) | 1 | ✅ merged+archived |
| 4 | advanced-indexing (take/take_along_axis/put/diagonal/argwhere/compress/choose/select/ravel_multi_index) | 1 | ✅ merged+archived |
| 5 | ufuncs-extras (around/gcd/lcm/sinc/nan_to_num/logaddexp/float_power/modf/frexp/unwrap/i0/...) | 1 | ✅ merged+archived |
| 6 | signal-poly (convolve/correlate/interp; polyval/polyfit/roots/poly/polyder/polyint) | 1 | ✅ merged+archived |
| 7 | sorting-extras (lexsort/sort_complex/searchsorted-sorter) | 1 | ✅ merged+archived |
| 8 | einsum (+tensordot/cross/cond/multi_dot) | 2 | ✅ merged+archived |
| 9 | random-distributions (full set + Philox/SFC64) | 2 | 🟡 next |
| 10 | polynomial-package (Polynomial/Chebyshev/...) | 2 | ⬜ |
| 11 | text-io (savetxt/loadtxt/genfromtxt/print-options) | 2 | ⬜ |
| 12 | char-strings (numpy.char) | 2 | ⬜ |

## Bug log (oracle divergences → issues)
| # | issue | capability | regression test | status |
|---|-------|-----------|-----------------|--------|
| 1 | [#26](https://github.com/CyberdyneCorp/NumPP/issues/26) spacing() sign wrong for negative inputs | ufuncs-extras | test_mathx.cpp `spacing vs numpy` | ✅ fixed (pre-merge) |
