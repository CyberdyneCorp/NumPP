# Add text & binary I/O (NumPy parity, Tier 2)

## Why
Eleventh item of the numpy-parity-roadmap backlog. NumPy's text table I/O,
raw binary I/O and integer string formatting were missing from NumPP.

## What changes
- **text-io** capability:
  - text tables: loadtxt, savetxt, genfromtxt (NaN for missing fields)
  - parsing: fromstring (separated numbers)
  - raw binary: tofile, fromfile
  - integer formatting: binary_repr (with two's-complement width), base_repr

## Non-goals
- structured-dtype / string-column loadtxt, converters/usecols/skiprows options,
  set_printoptions / array2string formatting controls (issue #11), `.npy` for
  structured dtype (issue #14) — deferred.
