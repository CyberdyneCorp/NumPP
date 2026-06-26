# Add vectorized strings (numpy.char) (NumPy parity, Tier 2)

## Why
Twelfth and final item of the numpy-parity-roadmap backlog. NumPy's vectorized
string operations (numpy.char) over string arrays were missing from NumPP.

## What changes
- **char-strings** capability (namespace `numpp::npchar`):
  - string -> string: add, multiply, upper, lower, capitalize, title,
    strip/lstrip/rstrip, replace
  - string -> int64/bool: str_len, find, count, startswith, endswith

## Non-goals
- split/rsplit/partition and join returning object/ragged arrays, encode/decode
  between 'U'/'S', center/ljust/rjust/zfill, unicode (non-ASCII) case folding,
  N-D string arrays (1-D supported) — deferred.
