# string-dtype Specification

## Purpose
TBD - created by archiving change add-portable-numpy-gaps. Update Purpose after archive.
## Requirements
### Requirement: Variable-length UTF-8 StringDType
NumPP SHALL provide a NumPy-2.0-style `StringDType`: an array dtype whose elements
are variable-length UTF-8 strings (not fixed-width 'U'/'S'). It SHALL support
creation from a list of strings, element get/set, equality/comparison, and the
core `numpy.char`-style operations needed for parity, with NA/missing handling
consistent with numpy's default (no missing value) configuration.

#### Scenario: round-trip and elementwise ops on variable-length strings
- GIVEN an array created from `["a", "bb", "ccc"]` with the variable-length StringDType
- THEN element access returns the original strings (no truncation to a fixed width)
- AND elementwise equality / concatenation match numpy's `StringDType` results

