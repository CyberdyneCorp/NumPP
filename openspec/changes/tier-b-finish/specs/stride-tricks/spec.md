# stride-tricks Specification

## MODIFIED Requirements

### Requirement: Stride tricks and functional helpers
NumPP SHALL provide sliding_window_view, as_strided, piecewise, apply_along_axis,
AND vectorize and apply_over_axes matching numpy.

#### Scenario: stride tricks match numpy
- WHEN sliding_window_view/as_strided/piecewise/apply_along_axis/vectorize/
  apply_over_axes are evaluated
- THEN the result equals the corresponding numpy function
