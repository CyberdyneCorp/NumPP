# ufunc-completion Specification

## ADDED Requirements

### Requirement: Rounding, type inspection and bit packing
NumPP SHALL provide fix, real_if_close, iscomplexobj, isrealobj, iscomplex,
isreal, common_type, packbits and unpackbits matching numpy.

#### Scenario: ufunc/type helpers match numpy
- WHEN fix/real_if_close/iscomplex/isreal/packbits/unpackbits are evaluated
- THEN the result equals the corresponding numpy function
- AND iscomplexobj/isrealobj/common_type report numpy's type classification

### Requirement: Complex-promoting math (emath)
NumPP SHALL provide emath::sqrt, emath::log and emath::power that promote to
complex for out-of-domain inputs, matching numpy.emath.

#### Scenario: emath matches numpy
- WHEN emath::sqrt/log/power are evaluated on inputs outside the real domain
- THEN the result equals numpy.emath.sqrt / numpy.emath.log / numpy.emath.power
