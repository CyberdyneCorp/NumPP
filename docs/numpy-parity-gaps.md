# NumPy → NumPP parity gaps & roadmap

Status of NumPP v1.0.0 against NumPy's public API, and a prioritized roadmap of
what's still missing. ✅ = implemented & oracle-validated, 🟡 = partial, ⬜ = missing.
Each ⬜ tier below is intended to become its own OpenSpec change.

## What's already done (v1.0.0)

| Area | Status |
|------|--------|
| ndarray: views, reshape/transpose/swapaxes/squeeze/expand_dims/ravel/flatten, broadcasting, basic+sliced indexing, copy/contiguity, astype | ✅ |
| dtypes: bool/int/uint/float16-64/complex64-128, NEP-50 promotion, casting; strings 'U'/'S', datetime64/timedelta64, structured | ✅ |
| creation: empty/zeros/ones/full(+_like), eye, identity, arange, linspace | ✅ |
| ufuncs: arithmetic/comparison/logical/bitwise/shift, trig+hyperbolic, exp/log family, sqrt/cbrt, floor/ceil/trunc/rint, hypot/arctan2/copysign, isnan/isinf/isfinite/signbit, abs/sign/conj/real/imag/angle, clip, where | ✅ |
| reductions: sum/prod/min/max/mean/std/var/any/all + nan{sum,mean,min,max,std,var} | ✅ |
| sorting/search: sort/argsort/partition/argpartition, searchsorted, unique(+idx/inv/counts), argmin/argmax, flatnonzero/count_nonzero/nonzero, in1d/isin/intersect1d/union1d/setdiff1d, bincount, histogram | ✅ |
| linalg: dot/vdot/inner/outer/trace/kron, solve/inv/det/slogdet/matrix_power, cholesky, qr, eigh/eigvalsh, eig/eigvals, svd/svdvals/pinv/matrix_rank/lstsq, norm | ✅ |
| fft: fft/ifft/rfft/irfft/hfft/ihfft, 2-D & N-D variants, fftfreq/rfftfreq/fftshift/ifftshift | ✅ |
| random: PCG64+MT19937, Generator+RandomState (bit-exact), normal/exponential/gamma/beta/chisquare/poisson/binomial, permutation/shuffle/choice | ✅ |
| I/O: .npy/.npz save/load, array_str/array_repr | ✅ |

---

## Tier 1 — high value, frequently used (do first)

### Array manipulation ⬜
`concatenate`, `stack`, `hstack`/`vstack`/`dstack`/`column_stack`,
`split`/`array_split`/`hsplit`/`vsplit`, `tile`, `repeat`, `flip`/`fliplr`/`flipud`,
`roll`, `rot90`, `moveaxis`/`rollaxis`, `atleast_1d`/`2d`/`3d`, `append`/`insert`/`delete`,
`pad`, `resize`, `trim_zeros`.

### Cumulative & differences ⬜
`cumsum`, `cumprod`, `nancumsum`/`nancumprod`, `diff`, `ediff1d`, `gradient`, `ptp`.

### Statistics ⬜
`median`, `percentile`, `quantile`, `average` (weighted), `cov`, `corrcoef`,
`digitize`, `nanmedian`/`nanpercentile`/`nanquantile`, `nanargmin`/`nanargmax`,
`histogram2d`/`histogramdd`.

### Creation & grids ⬜
`array`/`asarray` from nested data, `meshgrid`, `indices`, `diag`/`diagflat`,
`tri`/`tril`/`triu`, `vander`, `logspace`, `geomspace`, `fromfunction`,
`frombuffer`/`fromiter`.

### Advanced indexing ⬜
`take`, `put`, `take_along_axis`, `put_along_axis`, `diagonal`, `argwhere`,
`select`, `compress`, `choose`, `extract`/`place`, `ravel_multi_index`/`unravel_index`,
`ix_`, fancy (integer-array) and boolean indexing on `ndarray`.

### Remaining ufuncs ⬜
`round`/`around`, `fix`, `gcd`, `lcm`, `sinc`, `degrees`/`radians`, `nan_to_num`,
`logaddexp`/`logaddexp2`, `float_power`, `fmod`, `heaviside`, `modf`, `frexp`,
`ldexp`, `divmod`, `unwrap`, `i0`, `real_if_close`, `nextafter`, `spacing`.

### Signal / polynomial basics ⬜
`convolve`, `correlate`, `interp`; `polyval`, `polyfit`, `roots`, `poly`,
`polyadd`/`polysub`/`polymul`/`polydiv`, `polyder`, `polyint`, `poly1d`.

### Sorting extras ⬜
`lexsort`, `sort_complex`, `flip`-based reversed sort, `searchsorted(sorter=)`,
array-valued `kth` for partition.

---

## Tier 2 — important, larger scope

### einsum ⬜ (high value)
`einsum` with a subscript parser (sum/transpose/trace/matmul/outer/batched);
plus `tensordot`, `cross`, `cond`, `multi_dot`, `tensorsolve`, `tensorinv`.

### Full random distribution set ⬜
`geometric`, `hypergeometric`, `laplace`, `logistic`, `lognormal`, `pareto`,
`rayleigh`, `weibull`, `vonmises`, `wald`, `triangular`, `standard_cauchy`,
`standard_t`, `zipf`, `gumbel`, `f`, `noncentral_chisquare`/`noncentral_f`,
`negative_binomial`, `power`, `multinomial`, `multivariate_normal`, `dirichlet`,
`bytes`; `Philox` and `SFC64` BitGenerators.

### Polynomial package ⬜
`numpy.polynomial`: `Polynomial`, `Chebyshev`, `Legendre`, `Hermite`, `Laguerre`
with `fit`/`roots`/`deriv`/`integ`/arithmetic.

### Text & binary I/O ⬜
`savetxt`/`loadtxt`/`genfromtxt`, `fromstring`/`fromfile`/`tofile`,
`set_printoptions`/`array2string` options (precision/threshold/suppress),
scientific-notation printing (issue #11), `binary_repr`/`base_repr`,
structured-dtype `.npy` (issue #14).

### Vectorized strings ⬜
`numpy.char`: `add`, `multiply`, `upper`/`lower`/`strip`/`split`/`join`,
`find`/`replace`/`count`/`startswith`/`endswith`, `encode`/`decode`.

---

## Tier 3 — long tail / specialized

- Masked arrays (`numpy.ma`) — a large subsystem.
- `object` dtype, `recarray`, `numpy.testing` (`assert_allclose`, …).
- Datetime business-day functions (`busday_count`, `is_busday`, `busday_offset`)
  and unit conversion; more datetime/timedelta unit coverage.
- Real GPU backends (Metal/CUDA/Vulkan/OpenCL) behind the existing weak-vtable
  slots (the CPU-reference device proves the dispatch path today).
- Bit-exact long-tail already tracked: `choice(replace=False)` (#7), ziggurat
  distributions (#8), standalone `MT19937` stream (#9).

---

## Suggested order

1. **add-array-manipulation** (Tier 1: concatenate/stack/split/tile/repeat/flip/roll/pad/atleast_*)
2. **add-statistics** (Tier 1: cumsum/diff/gradient/median/percentile/quantile/average/cov/corrcoef/digitize + nan variants + nanargmin/max)
3. **add-creation-grids-indexing** (Tier 1: meshgrid/diag/tri*/vander/logspace + take/put/diagonal/argwhere/select/fancy+boolean indexing)
4. **add-ufuncs-extras** (Tier 1: round/gcd/lcm/sinc/degrees/nan_to_num/logaddexp/float_power/modf/frexp/ldexp/divmod/unwrap + convolve/correlate/interp)
5. **add-polynomials** (Tier 1/2: poly1d/polyval/polyfit/roots + numpy.polynomial package)
6. **add-einsum** (Tier 2)
7. **add-random-distributions** (Tier 2: full distribution set + Philox/SFC64)
8. **add-text-io** (Tier 2: savetxt/loadtxt/genfromtxt/print-options/scientific)
9. **add-char-strings** (Tier 2: numpy.char)
10. Tier 3 as demand dictates.
