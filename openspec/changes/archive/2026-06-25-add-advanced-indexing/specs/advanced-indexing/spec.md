# advanced-indexing Specification

## ADDED Requirements

### Requirement: Gather and scatter
NumPP SHALL provide take (flat and along an axis), take_along_axis and put
(in-place, flat C-order) matching numpy, including negative-index handling.

#### Scenario: take and take_along_axis match numpy
- WHEN take/take_along_axis are applied with integer indices
- THEN the result equals numpy.take / numpy.take_along_axis

### Requirement: Extraction and selection
NumPP SHALL provide diagonal, argwhere, compress, extract, choose and select
matching numpy.

#### Scenario: diagonal and argwhere match numpy
- WHEN diagonal (with offset) or argwhere is applied
- THEN the result equals numpy.diagonal / numpy.argwhere

#### Scenario: choose and select match numpy
- WHEN choose/select are evaluated with index/condition lists
- THEN the result equals numpy.choose / numpy.select (first matching condition wins)

### Requirement: Flat and multi-index conversion
NumPP SHALL provide ravel_multi_index and unravel_index matching numpy for
C-order layouts.

#### Scenario: index conversion round-trips
- WHEN ravel_multi_index / unravel_index are applied with given dims
- THEN the result equals numpy.ravel_multi_index / numpy.unravel_index
