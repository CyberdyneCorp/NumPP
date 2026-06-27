# signal-poly Specification

## ADDED Requirements

### Requirement: interp left/right/period options
`interp` SHALL accept `left` and `right` fill values for inputs outside `xp`, and
a `period` argument for periodic interpolation (in which case `left`/`right` are
ignored and the inputs are wrapped), matching `numpy.interp`.

#### Scenario: interp options match numpy
- WHEN `interp(x, xp, fp, left=L, right=R)` is evaluated for `x` outside `[xp[0], xp[-1]]`
- THEN out-of-range points use `L`/`R` and the result equals `numpy.interp`
- WHEN `interp(x, xp, fp, period=P)` is evaluated
- THEN the result equals `numpy.interp(..., period=P)` within tolerance
