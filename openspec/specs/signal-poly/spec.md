# signal-poly Specification

## Purpose
TBD - created by archiving change add-signal-poly. Update Purpose after archive.
## Requirements
### Requirement: 1-D signal routines
NumPP SHALL provide convolve, correlate (full/same/valid modes) and interp
(1-D linear interpolation) matching numpy.

#### Scenario: convolve and interp match numpy
- WHEN convolve/correlate are applied with a given mode, or interp is evaluated
- THEN the result equals numpy.convolve / numpy.correlate / numpy.interp

### Requirement: Polynomial value, arithmetic and calculus
NumPP SHALL provide polyval, polyadd, polysub, polymul, polydiv, polyder and
polyint matching numpy (coefficients highest power first).

#### Scenario: polynomial arithmetic matches numpy
- WHEN polyval/polyadd/polymul/polydiv/polyder/polyint are evaluated
- THEN the result equals the corresponding numpy.poly* function

### Requirement: Roots, coefficients and fitting
NumPP SHALL provide poly (from roots), roots (companion-matrix eigenvalues) and
polyfit (Vandermonde least squares) matching numpy.

#### Scenario: roots and polyfit match numpy
- WHEN roots is applied to coefficients (compared as sorted real parts)
- THEN it equals numpy.roots
- WHEN polyfit fits a polynomial of given degree
- THEN the coefficients equal numpy.polyfit

### Requirement: interp left/right/period options
`interp` SHALL accept `left` and `right` fill values for inputs outside `xp`, and
a `period` argument for periodic interpolation (in which case `left`/`right` are
ignored and the inputs are wrapped), matching `numpy.interp`.

#### Scenario: interp options match numpy
- WHEN `interp(x, xp, fp, left=L, right=R)` is evaluated for `x` outside `[xp[0], xp[-1]]`
- THEN out-of-range points use `L`/`R` and the result equals `numpy.interp`
- WHEN `interp(x, xp, fp, period=P)` is evaluated
- THEN the result equals `numpy.interp(..., period=P)` within tolerance

