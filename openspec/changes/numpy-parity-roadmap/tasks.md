# Backlog — NumPy parity (each becomes its own OpenSpec change when started)

## Tier 1
- [ ] add-array-manipulation: concatenate/stack/{h,v,d}stack/column_stack, split/array_split/{h,v}split, tile, repeat, flip/fliplr/flipud, roll, rot90, moveaxis/rollaxis, atleast_{1,2,3}d, append/insert/delete, pad, resize
- [ ] add-statistics: cumsum/cumprod/nancumsum/nancumprod, diff/ediff1d/gradient/ptp, median/percentile/quantile/average, cov/corrcoef, digitize, nanmedian/nanpercentile, nanargmin/nanargmax, histogram2d/histogramdd
- [ ] add-creation-grids: array/asarray (nested), meshgrid, indices, diag/diagflat, tri/tril/triu, vander, logspace, geomspace, fromfunction
- [ ] add-advanced-indexing: take/put, take_along_axis/put_along_axis, diagonal, argwhere, select, compress, choose, ravel_multi_index/unravel_index, ix_, fancy + boolean indexing
- [ ] add-ufuncs-extras: round/around, fix, gcd, lcm, sinc, degrees/radians, nan_to_num, logaddexp/logaddexp2, float_power, fmod, heaviside, modf, frexp, ldexp, divmod, unwrap, i0
- [ ] add-signal-poly: convolve, correlate, interp; polyval/polyfit/roots/poly/polyder/polyint/poly1d
- [ ] add-sorting-extras: lexsort, sort_complex, searchsorted(sorter=), array-kth partition

## Tier 2
- [ ] add-einsum: subscript parser + tensordot/cross/cond/multi_dot/tensorsolve/tensorinv
- [ ] add-random-distributions: geometric/hypergeometric/laplace/logistic/lognormal/pareto/rayleigh/weibull/vonmises/wald/triangular/standard_cauchy/standard_t/zipf/gumbel/f/noncentral*/negative_binomial/power/multinomial/multivariate_normal/dirichlet/bytes; Philox + SFC64
- [ ] add-polynomial-package: numpy.polynomial (Polynomial/Chebyshev/Legendre/Hermite/Laguerre)
- [ ] add-text-io: savetxt/loadtxt/genfromtxt, fromstring/fromfile/tofile, set_printoptions/array2string, scientific printing (#11), binary_repr/base_repr, structured .npy (#14)
- [ ] add-char-strings: numpy.char vectorized string ops

## Tier 3
- [ ] masked arrays (numpy.ma)
- [ ] object dtype, recarray, numpy.testing helpers
- [ ] datetime business-day + unit conversion
- [ ] real GPU backends (Metal/CUDA/Vulkan/OpenCL) behind the weak-vtable slots
- [ ] tracked bit-exact long-tail: #7 choice(replace=False), #8 ziggurat, #9 standalone MT19937
