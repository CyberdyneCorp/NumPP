# masked-arrays Specification

## ADDED Requirements

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
