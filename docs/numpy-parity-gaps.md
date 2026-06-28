# NumPy → NumPP parity gaps & roadmap

Status of NumPP against NumPy's public API, and a prioritized roadmap of
what's still missing. ✅ = implemented & oracle-validated, 🟡 = partial, ⬜ = missing.
Each ⬜ tier below is intended to become its own OpenSpec change.

> **Update (2026-06-27, v1.1.0):** **Tiers 1, 2, A, B and C are all delivered** —
> 45 baseline capabilities, **1861 oracle checks across 649 cases vs NumPy 2.1.3,
> 0 divergences**, plus **real OpenCL + CUDA GPU backends with tiled GEMM**
> validated on an RTX 5060, bit-exact RNG (PCG64/SFC64/Philox/MT19937 + ziggurat +
> `choice(replace=False)`), masked arrays, and DLPack + memmap interop. Bugs found
> & fixed across the build-out: #26 (spacing), legint(m=2), a CI `<algorithm>`
> include; bit-exact gaps #7/#8/#9/#36 all closed. The full roadmap to date is
> **complete**.
>
> What remains is the genuine NumPy long-tail, grouped below by *why* it's missing
> (§"What's left"). **Bucket C** (portable, no-dependency, charter-compatible) is
> now **fully delivered** (`add-portable-numpy-gaps`: errstate, memory-overlap,
> iteration API, array-API linalg aliases, busdaycalendar, einsum optimize, masked
> hard/soft masks, polynomial domain/window+fit, variable-length StringDType —
> **711 oracle cases / 1979 checks, 0 failures**). Only Buckets A (needs a Python
> runtime/object model) and B (needs an external dependency) remain, deferred by
> design.

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
`einsum` with a subscript parser (sum/transpose/trace/matmul/outer) + `optimize=`/
`einsum_path`; plus `tensordot`, `cross`, `cond`, `multi_dot`. (`tensorsolve`/
`tensorinv` and einsum ellipsis `...` are **not** yet shipped — see
`numpy-foundation-completion`.)

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
`noncentral_chisquare`, `hypergeometric`, `noncentral_f`. (`bytes` shipped too.)

### add-bitgenerators ✅
`SFC64`, `Philox`, standalone `MT19937` — all **bit-exact** with numpy (#9/#36
closed). PCG64 was already bit-exact.

### add-polynomial-classes ✅ (classes + calculus shipped)
The `Polynomial`/`Chebyshev`/`Legendre`/`Hermite`/`HermiteE`/`Laguerre` classes
with `roots`/`deriv`/`integ`/arithmetic/conversion, plus `*vander`/`*roots`/
`*companion`. Remaining (Bucket C below): explicit `domain`/`window` mapping and
class-level `fit`.

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
`is_busday`/`busday_count`/`busday_offset`, `datetime_as_string` ✅;
structured-dtype `.npy` (#14) **delivered v1.4.0**. Deferred: full datetime unit
conversion.

### add-gpu-backends ✅
**Real OpenCL + CUDA backends** behind the weak-vtable slots, validated on an
NVIDIA RTX 5060 (sm_120): float32/64 add/sub/mul/div/neg/sqrt + sum/prod +
**GEMM/matmul** on the GPU (CUDA via PTX-virtual JIT), plus the **v1.5.0 ScyPP
acceleration slots** (csr_spmv/cdist_euclidean/correlate1d, #99/#106). Deferred:
Metal/Vulkan (#135), CSR SpMM (#134), CSR-adaptive SpMV (#136), GPU hardware CI
(#135), device-resident buffers / async transfer, cuBLAS/clBLAST path.

### add-bitexact-longtail ✅
**Philox (#36)**, standalone **MT19937 (#9)**, `choice(replace=False)` (#7) and
the ziggurat `standard_normal`/`standard_exponential` (#8) are all now bit-exact
with numpy (joining PCG64 and SFC64). All four BitGenerators are bit-exact.

### add-interop-misc ✅ (the portable parts)
**DLPack** (`to_dlpack`/`from_dlpack`, zero-copy) and **`memmap`** (mmap-backed
arrays) shipped. Deferred (need external deps — see Bucket B): DEFLATE
`savez_compressed` (zlib), pocketfft/FFTW FFT backend, `ctypeslib`.

---

## What's left — the genuine NumPy long-tail, grouped by *why*

The whole roadmap above is delivered. What NumPP still lacks vs real NumPy falls
into three buckets, separated by the reason it's missing. This is the accurate,
current accounting (verified against the source, not against older notes).

### Bucket A — needs a Python runtime / dynamic object model (won't port cleanly)
| Missing | Note |
|---------|------|
| `object` dtype | type-erased Python-object arrays — no equivalent without a runtime object model |
| `recarray` | attribute-access record arrays (structured field *access* exists; the `.field` class doesn't) |
| `frompyfunc` + multi-arg `vectorize` | only the 1-arg scalar `vectorize` exists |
| `ctypeslib`, `np.matrix` | ctypes-specific / deprecated upstream — N/A |

### Bucket B — needs an external dependency (breaks the no-dep, iOS/Android charter)
| Missing | Note |
|---------|------|
| `savez_compressed` real DEFLATE | exists but stores *uncompressed* (numpy-readable); real compression needs zlib |
| FFTW / pocketfft FFT backend | optional accel dep |
| `longdouble` / `float128` | platform extended precision |
| Metal / Vulkan GPU | other-platform backends (OpenCL + CUDA shipped) |

### Bucket C — portable, charter-compatible ✅ DELIVERED (`add-portable-numpy-gaps`)
These were both missing *and* implementable in dependency-free C++. All nine
shipped — each its own oracle-validated PR, merged after CI green.

| Feature | What it is | Status |
|---------|------------|:------:|
| `errstate`/`seterr`/`geterr` | floating-point error-state control | ✅ |
| `shares_memory`/`may_share_memory` | memory-overlap detection between arrays | ✅ |
| `nditer`/`ndenumerate`/`ndindex` | public iterator API | ✅ |
| array-API 2023 names | `matrix_transpose`, `vecdot`, `vector_norm`, `matrix_norm`, `permute_dims` | ✅ |
| `busdaycalendar` | custom business-day calendars (`weekmask`/`holidays`) | ✅ |
| `einsum(optimize=)` / `einsum_path` | greedy contraction-order optimizer | ✅ |
| masked hard/soft masks | `harden_mask`/`soften_mask`/`hardmask` assignment semantics | ✅ |
| polynomial-class `domain`/`window` + `fit` | the remaining `numpy.polynomial` class surface | ✅ |
| NumPy-2.0 `StringDType` | variable-length UTF-8 strings (standalone container) | ✅ |

With Bucket C delivered, the only remaining NumPy surface is Buckets A and B —
deferred *by design* because they need a Python runtime/object model or an external
dependency, both of which break NumPP's clean-room, dependency-free, iOS/Android
charter.

### Foundation completion for a future C++ SciPy port (`numpy-foundation-completion`)
NumPP is intended as the **NumPy-equivalent base** that a separate C++ SciPy port
("SciPP") will build on. Boundary rule: `numpy.*` → NumPP, `scipy.*` → SciPP. The
`numpy-foundation-completion` change closed the genuine `numpy.*` primitives a SciPP
leans on — **all delivered ✅**:

| Tier | Item | Status |
|------|------|:------:|
| 1 | `finfo`/`iinfo` machine limits | ✅ |
| 1 | `isclose`, `isposinf`, `isneginf` | ✅ |
| 1 | `trapz`/`trapezoid` | ✅ |
| 1 | `promote_types`, `min_scalar_type` | ✅ |
| 2 | **batched/stacked linalg** (last-two-axes over solve/inv/det/slogdet/matrix_power/cholesky/qr/eig*/svd*/pinv/matrix_rank) | ✅ |
| 2 | `tensorsolve`/`tensorinv` | ✅ |
| 2 | `einsum` ellipsis `...` | ✅ |
| 2 | `interp` `left`/`right`/`period` | ✅ |

With this change delivered, **NumPP is a complete `numpy.*` foundation for a future
C++ SciPy port**. `lstsq` stays 2-D (matching numpy). The only remaining NumPy
surface is Buckets A and B, deferred by design.

`object` dtype, `recarray`, `frompyfunc` (Bucket A) are explicitly **out** — a C++
SciPP uses templates/variants/`std::function`, so adding them would be divergence
with no payoff. Metal stays a backend concern, not a `numpy.*` feature.

---

## Tracked open issues (backlog with full context)

Small, well-scoped gaps surfaced during NumPy-test mining and the ScyPP
acceleration work, each filed as a GitHub issue with where/repro/approach so the
work can resume cold. They also appear in the OpenSpec `numpy-parity-roadmap`
backlog. None block the `numpy.*` foundation; they are incremental parity/perf.

### `numpy.*` parity
| # | Gap |
|---|-----|
| #129 | complex `min`/`max` (amin/amax/minimum/maximum + reductions), lexicographic |
| #130 | `dot`/`inner`/`kron` for ndim>2 (N-D contraction / Kronecker) |
| #131 | `take(mode=wrap/clip)`, `ravel_multi_index`/`unravel_index(order='F')`, `repeat(per-element)` |
| #132 | `histogram(density=, weights=)` |
| #133 | `arange` integer precision beyond 2^53 |

### Device acceleration (shared `GpuVTable`, for ScyPP — OpenSpec `gpu-kernels`)
| # | Gap |
|---|-----|
| #134 | CSR **SpMM** (sparse × dense matrix) slot — follow-up to #99 |
| #135 | **Metal** backend + GPU **hardware CI** validation for the device kernels |
| #136 | vector/CSR-adaptive **SpMV** variant (performance) |
