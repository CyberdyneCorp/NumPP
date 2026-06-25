# hardening Specification

## Purpose
TBD - created by archiving change add-hardening. Update Purpose after archive.
## Requirements
### Requirement: NaN-ignoring reductions
NumPP SHALL provide nansum, nanmean, nanmin, nanmax, nanvar, and nanstd that
ignore NaN values, matching numpy, with axis/keepdims (and ddof for var/std).

#### Scenario: nansum ignores NaN
- WHEN nansum is applied to an array containing NaN
- THEN the result equals numpy.nansum (NaN treated as zero)

### Requirement: Zero-dependency portable build
The default build SHALL link with no required third-party dependency and the
CPU-only all-backends-off configuration SHALL configure successfully.

#### Scenario: No external deps
- WHEN the default library is built
- THEN it links without BLAS/LAPACK/GPU and the CPU-only config configures

