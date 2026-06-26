# masked-arrays Specification

## ADDED Requirements

### Requirement: Masked array core and reductions
NumPP SHALL provide a MaskedArray (data + bool mask), the constructors
masked_array/masked_where/masked_invalid/masked_equal/masked_greater, filled,
compressed, count and the reductions sum/mean/min/max over unmasked elements,
matching numpy.ma.

#### Scenario: masked operations match numpy.ma
- WHEN a masked array is built and filled/compressed/count are applied
- THEN the result equals the corresponding numpy.ma operation
- WHEN sum/mean/min/max reduce over the unmasked elements
- THEN the scalar equals numpy.ma's reduction ignoring masked entries
