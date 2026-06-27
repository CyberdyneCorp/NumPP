# einsum Specification

## ADDED Requirements

### Requirement: einsum contraction-order optimization
NumPP SHALL accept an `optimize` option on `einsum` (false, or a greedy
contraction-order search) that pairwise-reduces a multi-operand contraction, and
SHALL provide `einsum_path` returning the chosen contraction order. The numeric
result with `optimize` SHALL equal the unoptimized result (and numpy) within
floating-point tolerance.

#### Scenario: optimized multi-operand einsum matches the naive result
- GIVEN a 3+-operand einsum (e.g. a chained matrix product)
- WHEN evaluated with `optimize=true` and with `optimize=false`
- THEN both equal `numpy.einsum(...)` within tolerance
- AND `einsum_path` returns a valid pairwise contraction order
