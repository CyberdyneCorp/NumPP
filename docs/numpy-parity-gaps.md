# NumPy → NumPP parity gaps & roadmap

Status of NumPP v1.0.0 against NumPy's public API, and a prioritized roadmap of
what's still missing. ✅ = implemented & oracle-validated, 🟡 = partial, ⬜ = missing.
Each ⬜ tier below is intended to become its own OpenSpec change.

> **Update (2026-06-26):** the entire **Tier 1 + Tier 2 backlog is delivered**
> (12 capabilities, 28 baseline specs, 421 test cases / 1246 oracle checks, 0
> divergences; one bug found & fixed, #26). Items marked ✅ here may have a few
> advanced options deferred — see each change's "Non-goals" for specifics (e.g.
> Philox/SFC64 & discrete/multivariate random, the orthogonal-polynomial classes,
> fancy-indexing subscript operators). Only **Tier 3** remains.

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

### Array manipulation ✅
`concatenate`, `stack`, `hstack`/`vstack`/`dstack`/`column_stack`,
`split`/`array_split`/`hsplit`/`vsplit`, `tile`, `repeat`, `flip`/`fliplr`/`flipud`,
`roll`, `rot90`, `moveaxis`/`rollaxis`, `atleast_1d`/`2d`/`3d`, `append`/`insert`/`delete`,
`pad`, `resize`, `trim_zeros`.

### Cumulative & differences ✅
`cumsum`, `cumprod`, `nancumsum`/`nancumprod`, `diff`, `ediff1d`, `gradient`, `ptp`.

### Statistics ✅
`median`, `percentile`, `quantile`, `average` (weighted), `cov`, `corrcoef`,
`digitize`, `nanmedian`/`nanpercentile`/`nanquantile`, `nanargmin`/`nanargmax`,
`histogram2d`/`histogramdd`.

### Creation & grids ✅
`array`/`asarray` from nested data, `meshgrid`, `indices`, `diag`/`diagflat`,
`tri`/`tril`/`triu`, `vander`, `logspace`, `geomspace`, `fromfunction`,
`frombuffer`/`fromiter`.

### Advanced indexing ✅
`take`, `put`, `take_along_axis`, `put_along_axis`, `diagonal`, `argwhere`,
`select`, `compress`, `choose`, `extract`/`place`, `ravel_multi_index`/`unravel_index`,
`ix_`, fancy (integer-array) and boolean indexing on `ndarray`.

### Remaining ufuncs ✅
`round`/`around`, `fix`, `gcd`, `lcm`, `sinc`, `degrees`/`radians`, `nan_to_num`,
`logaddexp`/`logaddexp2`, `float_power`, `fmod`, `heaviside`, `modf`, `frexp`,
`ldexp`, `divmod`, `unwrap`, `i0`, `real_if_close`, `nextafter`, `spacing`.

### Signal / polynomial basics ✅
`convolve`, `correlate`, `interp`; `polyval`, `polyfit`, `roots`, `poly`,
`polyadd`/`polysub`/`polymul`/`polydiv`, `polyder`, `polyint`, `poly1d`.

### Sorting extras ✅
`lexsort`, `sort_complex`, `flip`-based reversed sort, `searchsorted(sorter=)`,
array-valued `kth` for partition.

---

## Tier 2 — important, larger scope

### einsum ✅ (high value)
`einsum` with a subscript parser (sum/transpose/trace/matmul/outer/batched);
plus `tensordot`, `cross`, `cond`, `multi_dot`, `tensorsolve`, `tensorinv`.

### Full random distribution set ✅
`geometric`, `hypergeometric`, `laplace`, `logistic`, `lognormal`, `pareto`,
`rayleigh`, `weibull`, `vonmises`, `wald`, `triangular`, `standard_cauchy`,
`standard_t`, `zipf`, `gumbel`, `f`, `noncentral_chisquare`/`noncentral_f`,
`negative_binomial`, `power`, `multinomial`, `multivariate_normal`, `dirichlet`,
`bytes`; `Philox` and `SFC64` BitGenerators.

### Polynomial package ✅
`numpy.polynomial`: `Polynomial`, `Chebyshev`, `Legendre`, `Hermite`, `Laguerre`
with `fit`/`roots`/`deriv`/`integ`/arithmetic.

### Text & binary I/O ✅
`savetxt`/`loadtxt`/`genfromtxt`, `fromstring`/`fromfile`/`tofile`,
`set_printoptions`/`array2string` options (precision/threshold/suppress),
scientific-notation printing (issue #11), `binary_repr`/`base_repr`,
structured-dtype `.npy` (issue #14).

### Vectorized strings ✅
`numpy.char`: `add`, `multiply`, `upper`/`lower`/`strip`/`split`/`join`,
`find`/`replace`/`count`/`startswith`/`endswith`, `encode`/`decode`.

---

# Remaining gaps roadmap (post Tier 1 + Tier 2)

The Tier 1 + Tier 2 sections above are now **delivered**. What follows is the
current, accurate accounting of what NumPP still lacks vs NumPy, regrouped into
three tiers. Each entry maps 1:1 to an unchecked item in the
`numpy-parity-roadmap` OpenSpec change (`tasks.md`) and graduates into its own
change when started.

## Tier A — complete partially-done modules ✅ DELIVERED (tier-a-completion)

### add-array-constructors ✅
`array`/`asarray`/`asanyarray` from nested initializer data, `fromiter`,
`frombuffer`, `fromregex`, `mgrid`/`ogrid`, `meshgrid(sparse=)`,
`broadcast_arrays`. (Closes the creation-grids non-goal.)

### add-manip-extras ✅
`block`, `dsplit`, `trim_zeros`, `rollaxis`, extra `pad` modes (reflect/
symmetric/wrap/linear_ramp/maximum/mean/median/minimum), `require`.

### add-indexing-completion ✅ (high value)
Integer-array and boolean **fancy indexing on the `ndarray` subscript operator**
(get *and* set), `put_along_axis`, `place`, `ix_`, `fill_diagonal`,
`diag_indices`, `tril_indices`/`triu_indices`, `mask_indices`, N-D `diagonal`.

### add-stats-extras ✅
`histogram2d`, `histogramdd`, `nanquantile`, weighted `cov`
(`fweights`/`aweights`), `gradient` with spacing/axis (N-D), `percentile`/
`quantile` `method=` interpolation options.

### add-ufunc-completion ✅
`fix`, `real_if_close`, `around` (negative decimals); type-check helpers
`iscomplexobj`/`isrealobj`/`isreal`/`iscomplex`/`isscalar`/`common_type`/
`mintypecode`; `numpy.emath` (scimath) complex-promoting `sqrt`/`log`/`power`;
`packbits`/`unpackbits`.

### add-poly1d ✅
`poly1d` class, complex-root reconstruction in `poly()`, weighted `polyfit`,
`polyvander`, `polycompanion`. (Closes the signal-poly non-goal.)

## Tier B — substantial new subsystems

### add-random-discrete-multivariate ⬜
`geometric`, `hypergeometric`, `zipf`, `logseries`, `negative_binomial`,
`multinomial`, `multivariate_normal`, `dirichlet`, `vonmises`, `wald`,
`standard_t`, `f`, `noncentral_chisquare`/`noncentral_f`, `bytes`.

### add-bitgenerators ⬜
`Philox`, `SFC64`, standalone bit-exact `MT19937` (#9), `SeedSequence` spawning,
`bit_generator.state` get/set.

### add-polynomial-classes ⬜
`Chebyshev`/`Legendre`/`Hermite`/`HermiteE`/`Laguerre` classes with domain/window
mapping, `fit`/`roots`/`deriv`/`integ`/conversion, plus `*fromroots`/`*vander`/
`*companion`/`*gauss`. (Completes the polynomial-package non-goal.)

### add-char-strings-completion ⬜
`split`/`rsplit`/`partition`/`splitlines`, `join`, `encode`/`decode`,
`center`/`ljust`/`rjust`/`zfill`, `expandtabs`, the `is*` predicates,
N-D string arrays, and NumPy 2.0 variable-width `StringDType`.

### add-printoptions ⬜
`set_printoptions`/`get_printoptions`/`printoptions`, `array2string` options
(precision/threshold/edgeitems/suppress/sign/floatmode), `format_float_positional`/
`format_float_scientific`, scientific printing (#11).

### add-stride-tricks ⬜
`sliding_window_view`, `as_strided`, `vectorize`, `frompyfunc`, `piecewise`,
`apply_along_axis`, `apply_over_axes`.

## Tier C — large / specialized subsystems

### add-masked-arrays ⬜
`numpy.ma` MaskedArray + masked ufuncs/reductions — a large subsystem.

### add-object-records-testing ⬜
`dtype=object` arrays, `recarray`/structured field access, and `numpy.testing`
(`assert_allclose`, `assert_array_equal`, …).

### add-datetime-completion ⬜
`busday_count`/`is_busday`/`busday_offset`, unit conversion, `datetime_as_string`,
more unit coverage, structured-dtype `.npy` (#14).

### add-gpu-backends ⬜
Real Metal/CUDA/Vulkan/OpenCL kernels behind the existing weak-vtable slots
(the CPU-reference device proves the dispatch path today); device residency /
async transfer.

### add-bitexact-longtail ⬜
The tracked bit-exact items: `choice(replace=False)` (#7), ziggurat normal/
exponential (#8), standalone `MT19937` stream (#9).

### add-interop-misc ⬜
`memmap`, `savez_compressed`, DLPack / array-API (`__dlpack__`/`from_dlpack`),
`ctypeslib`, and an optional pocketfft/FFTW FFT backend.

---

## Suggested order

Tier A first (each is small and closes a visible gap in an existing module), then
Tier B (new subsystems, largest user value: fancy indexing, the polynomial
classes, print options), then Tier C as demand dictates. **add-indexing-completion**
(fancy/boolean ndarray subscripting) is the highest-impact single item and is a
good next pick.
