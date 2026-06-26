# polynomial-package Specification

## ADDED Requirements

### Requirement: Power-basis polynomial API (lowest-first)
NumPP SHALL provide polyval, polyadd, polysub, polymul, polyder, polyint,
polyroots and polyfit in numpy.polynomial's lowest-degree-first convention.

#### Scenario: power-basis functions match numpy.polynomial
- WHEN any numpp::polynomial power-basis function is evaluated
- THEN the result equals the corresponding numpy.polynomial.polynomial function

### Requirement: Orthogonal basis evaluation
NumPP SHALL provide chebval, legval, hermval, hermeval and lagval matching
numpy's Chebyshev, Legendre, Hermite, HermiteE and Laguerre evaluation.

#### Scenario: orthogonal basis evaluation matches numpy
- WHEN chebval/legval/hermval/hermeval/lagval are evaluated
- THEN the result equals numpy.polynomial.<basis>.<basis>val

### Requirement: Polynomial class
NumPP SHALL provide a Polynomial class supporting evaluation, deriv, integ,
roots, addition, subtraction, multiplication and a static fit.

#### Scenario: Polynomial class matches numpy
- WHEN a Polynomial is evaluated, differentiated, multiplied or fit
- THEN the result equals numpy.polynomial.Polynomial for the same operation
