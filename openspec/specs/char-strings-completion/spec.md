# char-strings-completion Specification

## Purpose
TBD - created by archiving change tier-b-completion. Update Purpose after archive.
## Requirements
### Requirement: Additional vectorized string operations
NumPP SHALL provide center, ljust, rjust, zfill, swapcase, expandtabs and the
predicates isalpha, isdigit, isspace, isupper, islower, isalnum, istitle, AND
join, encode, decode and partition over string arrays, matching numpy.char.

#### Scenario: string ops and predicates match numpy.char
- WHEN a numpp::npchar string transformation or predicate is applied
- THEN the result equals the corresponding numpy.char function

#### Scenario: join, encode, decode, partition match numpy.char
- WHEN join/encode/decode/partition are applied to a string array
- THEN the result equals numpy.char.join / encode / decode / partition

