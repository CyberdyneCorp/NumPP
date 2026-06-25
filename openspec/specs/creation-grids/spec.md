# creation-grids Specification

## Purpose
TBD - created by archiving change add-creation-grids. Update Purpose after archive.
## Requirements
### Requirement: Coordinate grids
NumPP SHALL provide meshgrid (cartesian 'xy' default and matrix 'ij' indexing)
and indices, matching numpy.

#### Scenario: meshgrid matches numpy
- WHEN meshgrid is applied to coordinate vectors with a given indexing
- THEN each output equals the corresponding numpy.meshgrid output

### Requirement: Diagonals and triangles
NumPP SHALL provide diag (build from 1-D, extract from 2-D, with offset k),
diagflat, tri, tril and triu matching numpy.

#### Scenario: diag and triangles match numpy
- WHEN diag/diagflat/tri/tril/triu are applied with an offset k
- THEN the result equals numpy.diag / numpy.tri / numpy.tril / numpy.triu

### Requirement: Grid and matrix generators
NumPP SHALL provide vander (increasing/decreasing), logspace, geomspace and
fromfunction matching numpy.

#### Scenario: generators match numpy
- WHEN vander/logspace/geomspace/fromfunction are evaluated
- THEN the result equals the corresponding numpy function

