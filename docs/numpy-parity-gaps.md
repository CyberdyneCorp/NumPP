# NumPy → NumPP parity gaps & roadmap

Status of NumPP v1.0.0 against NumPy's public API, and a prioritized roadmap of
what's still missing. ✅ = implemented & oracle-validated, 🟡 = partial, ⬜ = missing.
Each ⬜ tier below is intended to become its own OpenSpec change.

> **Update (2026-06-26):** **Tiers 1, 2, A, B and most of C are delivered** — 44
> baseline capabilities, **626 test cases / 1779 oracle checks, 0 failures**, plus
> **real OpenCL + CUDA GPU backends (incl. GPU matmul)** validated on an RTX 5060.
> Bugs found & fixed: #26 (spacing), legint(m=2); open/tracked: #36 (Philox not
> bit-exact; SFC64 is). Items marked ✅ may have advanced options deferred (see
> each change's "Non-goals"). **Remaining backlog: just `add-bitexact-longtail`
> and `add-interop-misc`** (both deferred with rationale) plus assorted advanced
> sub-items.

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

## Tier B — substantial new subsystems ✅ MOSTLY DELIVERED (tier-b-completion)

### add-random-discrete-multivariate ✅
`geometric`, `zipf`, `logseries`, `negative_binomial`, `multinomial`,
`multivariate_normal`, `dirichlet`, `vonmises`, `wald`, `standard_t`, `f`,
`noncentral_chisquare`. Deferred: `hypergeometric`, `noncentral_f`, `bytes`.

### add-bitgenerators 🟡
`SFC64` ✅ (bit-exact). `Philox` 🟡 (deterministic; numpy bit-exactness tracked
in **#36**). Deferred: standalone bit-exact `MT19937` (#9), `SeedSequence`
spawning, `bit_generator.state` get/set.

### add-polynomial-classes ⬜ (only `*vander`/`*roots` shipped via orthopoly)
The `Chebyshev`/`Legendre`/`Hermite`/`HermiteE`/`Laguerre` **classes** with
domain/window mapping, `fit`/`roots`/`deriv`/`integ`/conversion, plus
`*fromroots`/`*companion`/`*gauss`. (`*vander` and `*roots` free functions are
done; the classes and calculus are the remaining work.)

### add-char-strings-completion ✅
`center`/`ljust`/`rjust`/`zfill`, `swapcase`, `expandtabs`, and the `is*`
predicates. Deferred: `split`/`rsplit`/`partition`/`splitlines`, `join`,
`encode`/`decode`, N-D string arrays, NumPy 2.0 `StringDType`.

### add-printoptions ✅
`set_printoptions`/`get_printoptions`, `format_float_positional`/
`format_float_scientific`, `array2string`. Deferred: full `array2string`
threshold/edgeitems/floatmode controls.

### add-stride-tricks ✅
`sliding_window_view`, `as_strided`, `piecewise`, `apply_along_axis` (+ `polyvander`,
`polycompanion`, `mask_indices`). Deferred: `vectorize`, `frompyfunc`, `apply_over_axes`.

## Tier C — large / specialized subsystems (mostly delivered)

### add-masked-arrays ✅
`numpy.ma` MaskedArray + masked_where/invalid/equal/greater, filled/compressed/
count, sum/mean/min/max. Deferred: arithmetic operators, per-axis reductions,
hard/soft masks.

### add-object-records-testing 🟡
`numpy.testing` (array_equal/array_equiv/assert_*) ✅; structured field access
already existed. Deferred: `dtype=object` arrays (type-erased storage), `recarray`.

### add-datetime-completion 🟡
`is_busday`/`busday_count`/`busday_offset`, `datetime_as_string` ✅. Deferred:
full unit conversion, structured-dtype `.npy` (#14).

### add-gpu-backends ✅
**Real OpenCL + CUDA backends** behind the weak-vtable slots, validated on an
NVIDIA RTX 5060 (sm_120): float32/64 add/sub/mul/div/neg/sqrt + sum/prod +
**GEMM/matmul** on the GPU (CUDA via PTX-virtual JIT). Deferred: Metal/Vulkan,
device-resident buffers / async transfer, tiled GEMM, Auto-routed matmul.

### add-bitexact-longtail ⬜
The tracked bit-exact items: `choice(replace=False)` (#7), ziggurat normal/
exponential (#8), standalone `MT19937` stream (#9), and **Philox (#36)** — all
deep numpy-internal reverse-engineering.

### add-interop-misc ⬜
`memmap`, DEFLATE `savez_compressed`, DLPack / array-API (`__dlpack__`/
`from_dlpack`), `ctypeslib`, and an optional pocketfft/FFTW FFT backend.

---

## What's left (post Tier A/B/C)

NumPP covers the full practical NumPy surface plus real GPU backends. Only two
backlog items remain unstarted, both deferred with rationale:
- **add-bitexact-longtail** — bit-exact parity for numpy's internal RNG
  algorithms (Philox #36, ziggurat #8, choice-without-replacement #7, MT19937 #9).
  Pure reverse-engineering, no deps; the hardest remaining work.
- **add-interop-misc** — memmap, compressed npz, DLPack/array-API, ctypeslib,
  FFTW. Mostly external dependencies that conflict with the no-dependency,
  iOS/Android-portable goal; pick individually as needed.

Plus the advanced sub-items deferred inside delivered changes (see each change's
Non-goals): masked-array operators, object dtype, Metal/Vulkan + tiled/device GEMM,
the polynomial-class domain/window + fit, char `split`/StringDType, `vectorize`/
`frompyfunc`, N-D gradient spacing, extra pad modes, `mask_indices`, etc.
