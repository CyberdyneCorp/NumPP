# orthopoly Specification

## ADDED Requirements

### Requirement: Orthogonal-basis Vandermonde and roots
NumPP SHALL provide chebvander, legvander, hermvander, hermevander, lagvander and
chebroots, legroots, hermroots, hermeroots, lagroots matching numpy.polynomial.

#### Scenario: vander and roots match numpy
- WHEN a basis Vandermonde matrix or the roots of a basis series are computed
- THEN the result equals numpy.polynomial.<basis>.<basis>vander /
  <basis>roots (roots compared as sorted real parts)
