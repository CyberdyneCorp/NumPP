# array-iteration Specification

## ADDED Requirements

### Requirement: Public iteration API
NumPP SHALL provide `ndindex(shape)` yielding every multi-index of a shape in C
order, `ndenumerate(a)` yielding `(index, value)` pairs, and a flat `nditer(a)`
that visits every element in C order regardless of the array's strides
(non-contiguous, broadcasted, transposed), matching numpy's visit order.

#### Scenario: iteration order matches numpy C-order traversal
- GIVEN an array `a` (including a transposed/non-contiguous view)
- WHEN iterated with `ndindex`/`ndenumerate`/`nditer`
- THEN the sequence of indices and the sequence of values match
  `numpy.ndindex` / `numpy.ndenumerate` / `numpy.nditer` over the same array
