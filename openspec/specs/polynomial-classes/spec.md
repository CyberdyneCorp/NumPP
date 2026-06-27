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

### Requirement: Domain/window mapping and class-level fit
The `numpy.polynomial` classes SHALL carry a `domain` and `window` and map inputs
from `domain` into `window` before evaluation (as numpy does), and SHALL provide a
class-level `fit(x, y, deg)` that least-squares fits in the mapped coordinates and
returns a fitted series whose `domain` defaults to the data range.

#### Scenario: fit and domain-mapped evaluation match numpy
- GIVEN sample points `(x, y)`
- WHEN a class `fit(x, y, deg)` is performed (e.g. `Chebyshev.fit`)
- THEN evaluating the result at `x` reproduces `y` within least-squares tolerance
- AND the coefficients and `domain` match `numpy.polynomial.*.fit` within tolerance

