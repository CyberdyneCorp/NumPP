# masked-arrays Specification

## Purpose
TBD - created by archiving change tier-c-partial. Update Purpose after archive.
## Requirements
### Requirement: Masked array core and reductions
NumPP SHALL provide a MaskedArray (data + bool mask), the constructors
masked_array/masked_where/masked_invalid/masked_equal/masked_greater, filled,
compressed, count and the reductions sum/mean/min/max over unmasked elements,
matching numpy.ma. It SHALL additionally provide elementwise arithmetic
(add/subtract/multiply/divide), the comparison/range constructors
(masked_less/masked_less_equal/masked_greater_equal/masked_not_equal/
masked_inside/masked_outside/masked_values), per-axis reductions
(sum/prod/mean/max/min/count along an axis) and the accessors getmask/getdata.

#### Scenario: masked operations match numpy.ma
- WHEN a masked array is built and filled/compressed/count are applied
- THEN the result equals the corresponding numpy.ma operation
- WHEN sum/mean/min/max reduce over the unmasked elements
- THEN the scalar equals numpy.ma's reduction ignoring masked entries

#### Scenario: masked arithmetic and per-axis reductions match numpy.ma
- WHEN add/subtract/multiply/divide combine two masked arrays
- THEN the data and mask (union) equal numpy.ma's result
- WHEN a per-axis reduction is applied
- THEN values equal numpy.ma's, and an output element is masked iff every element
  of its slice was masked

### Requirement: Hard and soft masks
NumPP `MaskedArray` SHALL support hard vs soft masks. `harden_mask` makes masked
entries un-assignable (assignment through a masked position is a no-op);
`soften_mask` restores the default where assigning an unmasked value also clears
the mask at that position. A `hardmask` query SHALL report the current mode,
matching numpy.ma.

#### Scenario: hard mask blocks assignment, soft mask clears it
- GIVEN a MaskedArray with a masked element
- WHEN the mask is hardened and a value is assigned to the masked position
- THEN the value and mask are unchanged
- WHEN the mask is softened and a value is assigned to the masked position
- THEN the value is written and that position becomes unmasked, matching numpy.ma

