# printoptions Specification

## ADDED Requirements

### Requirement: Float formatting and print options
NumPP SHALL provide format_float_positional, format_float_scientific,
set_printoptions/get_printoptions and array2string matching numpy's formatting.

#### Scenario: float formatting matches numpy
- WHEN format_float_positional/format_float_scientific format a value, or
  array2string renders an array with a given precision
- THEN the string equals numpy's output
- AND set_printoptions/get_printoptions round-trip the print configuration
