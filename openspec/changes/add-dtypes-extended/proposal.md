# Add extended dtypes (Phase 9)

## Why
NumPy supports string, datetime, and structured dtypes beyond the numeric set.
Phase 9 extends NumPP's dtype system to cover them.

## What changes
- **dtypes-extended** capability: extend core DType with a shared metadata
  side-channel (null for numeric dtypes); add String('U')/Bytes('S'),
  Datetime64/Timedelta64 (int64 + unit), and Struct (named field table) dtypes.
- (a) strings: creation, get/set, equality, printing, .npy descr round-trip.
- (b) datetime64/timedelta64: units, ISO parse/format, arithmetic.
- (c) structured: field table, a['name'] field views, record get/set.

## Non-goals
- Non-native byte order, object dtype, variable-length strings, full datetime
  business-day/timezone logic.
