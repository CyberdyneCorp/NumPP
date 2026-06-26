# char-strings-completion Specification

## ADDED Requirements

### Requirement: Additional vectorized string operations
NumPP SHALL provide center, ljust, rjust, zfill, swapcase, expandtabs and the
predicates isalpha, isdigit, isspace, isupper, islower, isalnum, istitle over
string arrays, matching numpy.char.

#### Scenario: string ops and predicates match numpy.char
- WHEN a numpp::npchar string transformation or predicate is applied
- THEN the result equals the corresponding numpy.char function
