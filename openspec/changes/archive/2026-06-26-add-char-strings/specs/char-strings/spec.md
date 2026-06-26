# char-strings Specification

## ADDED Requirements

### Requirement: Vectorized string transformations
NumPP SHALL provide add, multiply, upper, lower, capitalize, title, strip,
lstrip, rstrip and replace over 1-D string arrays, mirroring numpy.char.

#### Scenario: string transformations match numpy.char
- WHEN a numpp::npchar string-returning op is applied to a string array
- THEN each element equals the corresponding numpy.char result

### Requirement: Vectorized string queries
NumPP SHALL provide str_len, find, count, startswith and endswith over string
arrays, returning int64 / bool arrays matching numpy.char.

#### Scenario: string queries match numpy.char
- WHEN str_len/find/count/startswith/endswith are applied to a string array
- THEN the int64/bool result equals the corresponding numpy.char result
