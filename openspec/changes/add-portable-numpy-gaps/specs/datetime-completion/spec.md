# datetime-completion Specification

## ADDED Requirements

### Requirement: Custom business-day calendars
NumPP SHALL provide a `busdaycalendar(weekmask, holidays)` value and accept it
(or an inline `weekmask`/`holidays`) in `is_busday`, `busday_count` and
`busday_offset`, matching numpy: the weekmask selects valid weekdays and holidays
are excluded, with `roll` modes honored by `busday_offset`.

#### Scenario: weekmask and holidays change the business-day result
- GIVEN a calendar with a custom weekmask (e.g. Sun–Thu) and a holiday list
- WHEN `is_busday`/`busday_count`/`busday_offset` are evaluated over a date range
- THEN the results equal `numpy.busday_*` given the same `busdaycalendar`
