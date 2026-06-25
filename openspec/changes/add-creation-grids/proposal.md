# Add creation grids & matrix builders (NumPy parity, Tier 1)

## Why
Third item of the numpy-parity-roadmap backlog. NumPy's coordinate-grid and
matrix-builder routines (meshgrid, indices, diag, tri/tril/triu, vander,
logspace/geomspace, fromfunction) are widely used and were missing from NumPP.

## What changes
- **creation-grids** capability:
  - coordinate grids: meshgrid ('xy'/'ij'), indices
  - diagonals & triangles: diag (build/extract), diagflat, tri, tril, triu
  - generators: vander (increasing/decreasing), logspace, geomspace, fromfunction

## Non-goals
- array/asarray from nested initializer data, fromiter/frombuffer (deferred);
  N-D meshgrid `sparse=`, `copy=` options; mgrid/ogrid index-trick objects.
