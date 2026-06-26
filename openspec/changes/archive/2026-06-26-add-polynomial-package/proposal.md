# Add the polynomial package (NumPy parity, Tier 2)

## Why
Tenth item of the numpy-parity-roadmap backlog. numpy.polynomial provides the
modern lowest-degree-first power-basis API plus orthogonal polynomial bases;
none of it existed in NumPP.

## What changes
- **polynomial-package** capability (namespace `numpp::polynomial`):
  - power basis (lowest-first): polyval, polyadd, polysub, polymul, polyder,
    polyint, polyroots, polyfit
  - orthogonal basis evaluation via 3-term recurrences: chebval, legval,
    hermval (physicists'), hermeval (probabilists'), lagval
  - Polynomial class: call/deriv/integ/roots/+/-/*/fit

## Non-goals
- The Chebyshev/Legendre/Hermite/Laguerre classes with domain/window mapping,
  basis fit()/roots()/conversion, and *fromroots/*vander helpers (deferred).
