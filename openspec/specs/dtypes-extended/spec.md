# dtypes-extended Specification

## Purpose
TBD - created by archiving change add-dtypes-extended. Update Purpose after archive.
## Requirements
### Requirement: String and bytes dtypes
NumPP SHALL provide fixed-width unicode ('U') and bytes ('S') dtypes with array
creation from strings, element get/set as UTF-8 std::string, element-wise
equality, numpy-compatible printing, and .npy descr round-trip.

#### Scenario: Create and read a string array
- WHEN a string array is created from {"ab","c"} with width 2
- THEN element 0 reads back "ab", its dtype kind is 'U', and itemsize is 8

#### Scenario: repr matches numpy
- WHEN array_repr is computed for a 'U2' string array
- THEN it equals numpy's repr including the dtype='<U2' suffix

#### Scenario: npy interop
- WHEN a string array is saved and loaded
- THEN values round-trip, and numpy.load reads the same values

### Requirement: Datetime64 and timedelta64
NumPP SHALL provide datetime64/timedelta64 dtypes (int64 + unit) with ISO parse/
format and arithmetic (datetime-datetime=timedelta, datetime+timedelta=datetime).

#### Scenario: Datetime difference
- WHEN two day-unit datetimes are subtracted
- THEN the result is a timedelta of the correct number of days

### Requirement: Structured dtypes
NumPP SHALL provide structured dtypes (named fields with offsets) with field
access returning a strided view and record get/set.

#### Scenario: Field view
- WHEN a structured array's field is accessed by name
- THEN it returns a view of that field's dtype with the correct values

