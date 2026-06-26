# stride-tricks Specification

## Purpose
TBD - created by archiving change tier-b-completion. Update Purpose after archive.
## Requirements
### Requirement: Stride tricks and functional helpers
NumPP SHALL provide sliding_window_view, as_strided, piecewise, apply_along_axis,
AND vectorize and apply_over_axes matching numpy.

#### Scenario: stride tricks match numpy
- WHEN sliding_window_view/as_strided/piecewise/apply_along_axis/vectorize/
  apply_over_axes are evaluated
- THEN the result equals the corresponding numpy function

### Requirement: Polynomial and mask index helpers
NumPP SHALL provide polyvander, polycompanion and mask_indices matching numpy.

#### Scenario: helpers match numpy
- WHEN polyvander/polycompanion/mask_indices are evaluated
- THEN the result equals numpy.polynomial.polynomial.polyvander /
  polycompanion / numpy.mask_indices

