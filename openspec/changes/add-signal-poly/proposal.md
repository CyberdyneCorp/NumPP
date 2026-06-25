# Add signal & polynomial basics (NumPy parity, Tier 1)

## Why
Sixth item of the numpy-parity-roadmap backlog. NumPy's 1-D signal routines
(convolve/correlate/interp) and the legacy polynomial API (polyval/polyfit/
roots/poly + arithmetic/calculus) are widely used and were missing.

## What changes
- **signal-poly** capability:
  - signal: convolve, correlate, interp (full/same/valid modes; 1-D linear interp)
  - polynomial value/arithmetic: polyval, polyadd, polysub, polymul, polydiv
  - polynomial calculus: polyder, polyint
  - roots<->coeffs: poly (from roots), roots (companion-matrix eigenvalues)
  - fitting: polyfit (Vandermonde least squares)

## Non-goals
- poly1d class, complex-root reconstruction in poly(), 2-D convolution,
  weighted polyfit, numpy.polynomial package (Tier 2, separate change).
