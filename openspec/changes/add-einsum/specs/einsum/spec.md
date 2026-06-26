# einsum Specification

## ADDED Requirements

### Requirement: Einstein summation
NumPP SHALL provide einsum with a general subscript parser supporting implicit
and explicit output, repeated indices and summed indices, matching numpy for
matmul, trace, diagonal, transpose, sum, outer, inner and batched contractions.

#### Scenario: einsum matches numpy
- WHEN einsum is evaluated for a subscript over one or more operands
- THEN the result equals numpy.einsum for the same subscript

### Requirement: Tensor contraction helpers
NumPP SHALL provide tensordot (axis count or explicit axis lists), cross (last
axis length 2 or 3), cond (2-norm condition number) and multi_dot, matching numpy.

#### Scenario: tensordot and cross match numpy
- WHEN tensordot/cross are evaluated
- THEN the result equals numpy.tensordot / numpy.cross

#### Scenario: cond and multi_dot match numpy
- WHEN cond/multi_dot are evaluated
- THEN the result equals numpy.linalg.cond / numpy.linalg.multi_dot
