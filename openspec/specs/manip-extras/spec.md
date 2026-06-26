# manip-extras Specification

## Purpose
TBD - created by archiving change tier-a-completion. Update Purpose after archive.
## Requirements
### Requirement: Additional array manipulation
NumPP SHALL provide block, dsplit, trim_zeros and rollaxis matching numpy.

#### Scenario: manipulation extras match numpy
- WHEN block assembles nested blocks, dsplit splits along axis 2, trim_zeros
  trims a 1-D array, or rollaxis moves an axis
- THEN the result equals numpy.block / numpy.dsplit / numpy.trim_zeros /
  numpy.rollaxis

