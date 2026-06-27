# einsum Specification

## ADDED Requirements

### Requirement: einsum ellipsis subscripts
`einsum` SHALL accept the ellipsis (`...`) in subscripts to represent and
broadcast leading (batch) dimensions, matching numpy — e.g. `"...ij,...jk->...ik"`
for batched matrix multiply and `"...ii->...i"` for batched diagonals.

#### Scenario: ellipsis broadcasting matches numpy
- GIVEN operands with leading batch axes captured by `...`
- WHEN an ellipsis einsum (e.g. `"...ij,...jk->...ik"`) is evaluated
- THEN the result equals `numpy.einsum` with the same subscripts, including the
  broadcast of the leading dimensions
