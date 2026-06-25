# Add extra math ufuncs (NumPy parity, Tier 1)

## Why
Fifth item of the numpy-parity-roadmap backlog. A batch of commonly-used NumPy
math functions (rounding, angle conversion, integer gcd/lcm, log-sum-exp,
mantissa/exponent decompositions, phase unwrap, Bessel i0, …) were missing.

## What changes
- **ufuncs-extras** capability:
  - rounding/angle: around, degrees, radians, sinc
  - integer: gcd, lcm
  - nan/inf: nan_to_num
  - log-sum-exp / powers / mod: logaddexp, logaddexp2, float_power, fmod, heaviside
  - decompositions: modf, frexp, ldexp, divmod
  - misc: unwrap, i0, nextafter, spacing

## Non-goals
- fix (== trunc), real_if_close, around with negative decimals edge cases,
  complex-input variants of the misc functions (deferred).
