# array-constructors Specification

## Purpose
TBD - created by archiving change tier-a-completion. Update Purpose after archive.
## Requirements
### Requirement: Array construction from data and buffers
NumPP SHALL provide fromiter, frombuffer, broadcast_arrays, meshgrid_sparse,
mgrid and ogrid matching numpy.

#### Scenario: constructors match numpy
- WHEN fromiter/frombuffer build an array, or broadcast_arrays/meshgrid_sparse/
  mgrid/ogrid produce coordinate grids
- THEN the result equals the corresponding numpy function (numpy.fromiter,
  numpy.frombuffer, numpy.broadcast_arrays, numpy.meshgrid(sparse=True),
  numpy.mgrid, numpy.ogrid)

