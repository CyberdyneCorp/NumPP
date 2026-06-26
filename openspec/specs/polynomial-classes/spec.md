# polynomial-classes Specification

## Purpose
TBD - created by archiving change tier-b-finish. Update Purpose after archive.
## Requirements
### Requirement: Orthogonal-basis calculus
NumPP SHALL provide chebder/chebint, legder/legint, hermder/hermint,
hermeder/hermeint and lagder/lagint matching numpy.polynomial (scalar integration
constant applied to the first integration only, like numpy's padded k list).

#### Scenario: der/int match numpy
- WHEN a basis derivative or antiderivative (order m, constant k) is computed
- THEN the result equals numpy.polynomial.<basis>.<basis>der / <basis>int

### Requirement: Orthogonal-polynomial classes
NumPP SHALL provide Chebyshev, Legendre, Hermite, HermiteE and Laguerre classes
(default identity domain) supporting evaluation, deriv, integ and roots.

#### Scenario: classes match numpy default-domain classes
- WHEN a class is evaluated, differentiated, integrated, or its roots taken
- THEN the result equals the corresponding numpy.polynomial.<Class> with the
  default domain

