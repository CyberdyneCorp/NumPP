# Add array manipulation (NumPy parity, Tier 1)

## Why
First item of the numpy-parity-roadmap backlog: NumPy's array-manipulation
routines (joining, splitting, tiling, rearranging, editing) are widely used and
were missing from NumPP.

## What changes
- **array-manipulation** capability: concatenate/stack/hstack/vstack/dstack/
  column_stack, split/array_split/hsplit/vsplit, tile/repeat, flip/fliplr/flipud/
  roll/rot90/moveaxis/atleast_{1,2,3}d, append/insert/delete_/resize/pad.

## Non-goals
- `block`, `dsplit`, structured/record manipulation, pad modes beyond
  constant/edge (deferred).
