# Backlog — NumPy parity (each unchecked item becomes its own OpenSpec change)

## Delivered — Tier 1 (all merged + archived)
- [x] add-array-manipulation: concatenate/stack/{h,v,d}stack/column_stack, split/array_split/{h,v}split, tile, repeat, flip/fliplr/flipud, roll, rot90, moveaxis, atleast_{1,2,3}d, append/insert/delete, pad (constant/edge), resize
- [x] add-statistics: cumsum/cumprod/nancumsum/nancumprod, diff/ediff1d/gradient/ptp, median/percentile/quantile/average, cov/corrcoef, digitize, nanmedian/nanpercentile, nanargmin/nanargmax
- [x] add-creation-grids: meshgrid, indices, diag/diagflat, tri/tril/triu, vander, logspace, geomspace, fromfunction
- [x] add-advanced-indexing: take/take_along_axis/put, diagonal, argwhere, compress, extract, choose, select, ravel_multi_index/unravel_index
- [x] add-ufuncs-extras: around, degrees/radians, sinc, gcd, lcm, nan_to_num, logaddexp/logaddexp2, float_power, fmod, heaviside, modf, frexp, ldexp, divmod, unwrap, i0, nextafter, spacing
- [x] add-signal-poly: convolve, correlate, interp; polyval/polyadd/polysub/polymul/polydiv/polyder/polyint/poly/roots/polyfit
- [x] add-sorting-extras: lexsort, sort_complex, searchsorted(sorter=)

## Delivered — Tier 2 (all merged + archived)
- [x] add-einsum: subscript parser + tensordot/cross/cond/multi_dot
- [x] add-random-distributions: laplace/logistic/gumbel/rayleigh/weibull/pareto/power/standard_cauchy/triangular/lognormal (statistical validation)
- [x] add-polynomial-package: polynomial.* power basis + chebval/legval/hermval/hermeval/lagval + Polynomial class
- [x] add-text-io: loadtxt/savetxt/genfromtxt, fromstring, tofile/fromfile, binary_repr/base_repr
- [x] add-char-strings: numpy.char add/multiply/upper/lower/capitalize/title/strip*/replace/str_len/find/count/startswith/endswith

---

## Delivered — Tier A (merged via the `tier-a-completion` change)
> Headline functions of each item shipped; a few advanced sub-items (extra pad
> modes, N-D gradient spacing, `mask_indices`, `polyvander`, complex-root `poly()`,
> `fromregex`, `around` negative decimals) are deferred — see the change Non-goals.
- [x] add-array-constructors: array/asarray/asanyarray from nested data, fromiter, frombuffer, fromregex, mgrid/ogrid, meshgrid(sparse=), broadcast_arrays
- [x] add-manip-extras: block, dsplit, trim_zeros, rollaxis, extra pad modes (reflect/symmetric/wrap/linear_ramp/maximum/mean/median/minimum), require
- [x] add-indexing-completion: integer-array + boolean **fancy indexing on ndarray subscript** (get & set), put_along_axis, place, ix_, fill_diagonal, diag_indices, tril_indices/triu_indices, mask_indices, N-D diagonal
- [x] add-stats-extras: histogram2d, histogramdd, nanquantile, weighted cov (fweights/aweights), gradient with spacing/axis (N-D), percentile/quantile method= options
- [x] add-ufunc-completion: fix, real_if_close, around(negative decimals); type-check helpers iscomplexobj/isrealobj/isreal/iscomplex/isscalar/common_type/mintypecode; emath (scimath) complex-promoting sqrt/log/power; packbits/unpackbits
- [x] add-poly1d: poly1d class, complex-root poly(), weighted polyfit, polyvander/polycompanion

## Remaining — Tier A leftovers (deferred sub-items from tier-a-completion)
- [x] add-tier-a-leftovers: extra pad modes (reflect/symmetric/wrap/linear_ramp/maximum/mean/median/minimum), N-D gradient with spacing/axis, percentile/quantile method= options, mask_indices, N-D diagonal, polyvander/polycompanion, complex-root poly(), fromregex, asanyarray, around(negative decimals), isscalar/mintypecode

## Remaining — Tier B: substantial new subsystems
- [x] add-random-discrete-multivariate: geometric, hypergeometric, zipf, logseries, negative_binomial, multinomial, multivariate_normal, dirichlet, vonmises, wald, standard_t, f, noncentral_chisquare/noncentral_f, bytes
- [x] add-bitgenerators: Philox, SFC64, standalone bit-exact MT19937 (#9), SeedSequence spawning, bit_generator.state get/set
- [x] add-polynomial-classes: Chebyshev/Legendre/Hermite/HermiteE/Laguerre classes with domain/window, fit/roots/deriv/integ/convert + *fromroots/*vander/*companion/*gauss
- [x] add-char-strings-completion: split/rsplit/partition/splitlines, join, encode/decode, center/ljust/rjust/zfill, expandtabs, isalpha/isdigit/isspace predicates, N-D string arrays, StringDType (NumPy 2.0)
- [x] add-printoptions: set_printoptions/get_printoptions/printoptions, array2string options (precision/threshold/edgeitems/suppress/sign/floatmode), format_float_positional/scientific (#11)
- [x] add-stride-tricks: sliding_window_view, as_strided, vectorize, frompyfunc, piecewise, apply_along_axis, apply_over_axes

## Tier C — large / specialized subsystems (achievable subset delivered via tier-c-partial)
- [x] add-masked-arrays: numpy.ma MaskedArray + constructors, filled/compressed/count, flat AND per-axis sum/prod/mean/min/max/count, arithmetic (add/sub/mul/div), comparison/range masks, getmask/getdata. Deferred: hard/soft masks, masked records
- [x] add-object-records-testing: numpy.testing (array_equal/array_equiv/assert_*) ✅; structured field access already existed; **dtype=object arrays deferred** (type-erased storage)
- [x] add-datetime-completion: is_busday/busday_count/busday_offset, datetime_as_string ✅; full unit conversion & structured-dtype .npy (#14) deferred
- [x] add-gpu-backends: **real OpenCL + CUDA backends incl. GPU GEMM/matmul, a device buffer pool and a tiled shared-memory GEMM** (validated on NVIDIA RTX 5060 / sm_120; ~5-8x over CPU on fp64 matmul, 109 GFLOP/s @ 1024³; CUDA uses PTX-virtual arch so nvcc 12.0 JITs to sm_120). Deferred: Metal/Vulkan, true device-resident ndarrays, pinned/unified memory, cuBLAS/clBLAST path
- [x] add-bitexact-longtail: **COMPLETE** — Philox (#36), MT19937 (#9), ziggurat standard_normal/exponential (#8) and choice(replace=False) (#7) all bit-exact with numpy. (Random module now matches numpy across PCG64, SFC64, Philox, MT19937, ziggurat normals/exponentials, normal, and choice-without-replacement.)
- [ ] add-interop-misc: memmap, DEFLATE savez_compressed, DLPack/array-API, ctypeslib, FFTW — **deferred: external deps / conflict with no-dependency portability goal**
