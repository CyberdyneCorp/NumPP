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

### Requirement: Custom business-day calendars
NumPP SHALL provide a `busdaycalendar(weekmask, holidays)` value and accept it
(or an inline `weekmask`/`holidays`) in `is_busday`, `busday_count` and
`busday_offset`, matching numpy: the weekmask selects valid weekdays and holidays
are excluded, with `roll` modes honored by `busday_offset`.

#### Scenario: weekmask and holidays change the business-day result
- GIVEN a calendar with a custom weekmask (e.g. Sun–Thu) and a holiday list
- WHEN `is_busday`/`busday_count`/`busday_offset` are evaluated over a date range
- THEN the results equal `numpy.busday_*` given the same `busdaycalendar`

