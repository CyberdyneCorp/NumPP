# datetime-completion Specification

## Purpose
TBD - created by archiving change tier-c-partial. Update Purpose after archive.
## Requirements
### Requirement: Business-day calendar functions
NumPP SHALL provide is_busday, busday_count, busday_offset and datetime_as_string
over datetime64[D] arrays (Mon-Fri week), matching numpy.

#### Scenario: business-day functions match numpy
- WHEN is_busday/busday_count/busday_offset are evaluated over dates
- THEN the result equals numpy.is_busday / numpy.busday_count / numpy.busday_offset
- AND datetime_as_string produces numpy's ISO-8601 formatting

