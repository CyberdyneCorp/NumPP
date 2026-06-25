# Add I/O (Phase 7)

## Why

Phase 7 adds persistence and human-readable output: `.npy`/`.npz` save/load
matching numpy's on-disk format, and numpy-compatible array printing.

## What changes

- **io** capability: `save`/`load` for the NPY format (v1.0/v2.0 read, v1.0
  write; all dtypes, C/F order); `savez`/`savez_compressed`/`load` for NPZ
  (a ZIP of NPY members); and `array_str`/`array_repr`/`to_string` reproducing
  numpy's element formatting, summarization, and `array([...])` repr.

## Reuse vs rewrite

- The NPY reader replaces the test-only loader in `tests/oracle.hpp`, so the
  production reader is exercised by every oracle test.
- Minimal self-contained ZIP writer/reader for NPZ (zlib used for compression
  when available, else stored).

## Non-goals

- Pickled object arrays, memory-mapped load, structured-dtype `.npy` (deferred
  to the structured-dtype phase).
