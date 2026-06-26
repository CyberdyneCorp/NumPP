# text-io Specification

## ADDED Requirements

### Requirement: Text table I/O
NumPP SHALL provide loadtxt, savetxt and genfromtxt matching numpy for float
tables (whitespace or delimiter separated, '#' comments, 1-D when single
row/column), with genfromtxt filling missing fields with NaN.

#### Scenario: loadtxt / savetxt round-trip with numpy
- WHEN savetxt writes an array and numpy.loadtxt reads it (or vice versa)
- THEN the values are preserved and equal the original array

### Requirement: Parsing and raw binary I/O
NumPP SHALL provide fromstring (separated numbers), tofile and fromfile
matching numpy's raw little-endian binary layout.

#### Scenario: fromfile reads numpy tofile output
- WHEN numpy writes an array with ndarray.tofile and fromfile reads it back
- THEN the values equal the original array

### Requirement: Integer string formatting
NumPP SHALL provide binary_repr (with two's-complement width) and base_repr
matching numpy.

#### Scenario: binary_repr and base_repr match numpy
- WHEN binary_repr/base_repr are evaluated (including negative and padded cases)
- THEN the string equals numpy.binary_repr / numpy.base_repr
