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
| 9 | random-distributions (laplace/logistic/gumbel/rayleigh/weibull/pareto/power/cauchy/triangular/lognormal) | 2 | ✅ merged+archived |
| 10 | polynomial-package (polyval/fit/roots + cheb/leg/herm/herme/lag val + Polynomial class) | 2 | ✅ merged+archived |
| 11 | text-io (savetxt/loadtxt/genfromtxt/fromstring/tofile/fromfile/binary_repr/base_repr) | 2 | ✅ merged+archived |
| 12 | char-strings (numpy.char: add/multiply/upper/lower/strip/replace/find/count/startswith/endswith) | 2 | ✅ merged+archived |

**Loop complete — all 12 Tier 1 + Tier 2 backlog capabilities delivered.** Suite: 421 cases / 1246 checks, 0 divergences. 1 bug found+fixed (#26).

## Bug log (oracle divergences → issues)
| # | issue | capability | regression test | status |
|---|-------|-----------|-----------------|--------|
| 2 | [#36](https://github.com/CyberdyneCorp/NumPP/issues/36) Philox raw stream not bit-exact with numpy | bitgenerators | test_bitgenerators.cpp `Philox random_raw bit-exact vs numpy` | ✅ fixed (M0=0xD2E7470EE14C6C93 + increment-before-generate) |
| 1 | [#26](https://github.com/CyberdyneCorp/NumPP/issues/26) spacing() sign wrong for negative inputs | ufuncs-extras | test_mathx.cpp `spacing vs numpy` | ✅ fixed (pre-merge) |
