# poly1d Specification

## Purpose
TBD - created by archiving change tier-a-completion. Update Purpose after archive.
## Requirements
### Requirement: poly1d class and weighted fit
NumPP SHALL provide a poly1d class (highest-degree-first) supporting evaluation,
deriv, integ, roots and +/-/* arithmetic, plus polyfit_weighted, matching numpy.

#### Scenario: poly1d matches numpy
- WHEN a poly1d is evaluated, differentiated, integrated, multiplied, or its
  roots taken
- THEN the result equals numpy.poly1d for the same operation

#### Scenario: weighted polyfit matches numpy
- WHEN polyfit_weighted fits a polynomial with per-point weights
- THEN the coefficients equal numpy.polyfit(x, y, deg, w=w)

