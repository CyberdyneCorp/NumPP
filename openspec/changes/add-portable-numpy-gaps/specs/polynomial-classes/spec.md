# polynomial-classes Specification

## ADDED Requirements

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
