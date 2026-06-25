# ufuncs-extras Specification

## Purpose
TBD - created by archiving change add-ufuncs-extras. Update Purpose after archive.
## Requirements
### Requirement: Rounding, angle and integer ufuncs
NumPP SHALL provide around (round half to even), degrees, radians, sinc, gcd and
lcm matching numpy.

#### Scenario: around and sinc match numpy
- WHEN around(decimals)/sinc are applied
- THEN the result equals numpy.around / numpy.sinc

### Requirement: nan handling, log-sum-exp, powers and mod
NumPP SHALL provide nan_to_num, logaddexp, logaddexp2, float_power, fmod and
heaviside matching numpy.

#### Scenario: logaddexp and float_power match numpy
- WHEN logaddexp/logaddexp2/float_power/fmod/heaviside are evaluated
- THEN the result equals the corresponding numpy function

### Requirement: Decompositions and floating-point neighbours
NumPP SHALL provide modf, frexp, ldexp, divmod, unwrap, i0, nextafter and spacing
matching numpy.

#### Scenario: decompositions match numpy
- WHEN modf/frexp/divmod are evaluated
- THEN both returned arrays equal the corresponding numpy outputs

#### Scenario: spacing matches numpy sign convention
- WHEN spacing is evaluated for positive and negative inputs
- THEN the result steps away from zero, equal to numpy.spacing (including sign)

