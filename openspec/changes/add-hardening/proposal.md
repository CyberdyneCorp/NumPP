# Hardening + v1.0 (Phase 11)

## Why
Final phase: close cheap parity gaps, reduce the worst complexity hotspot,
document the library comprehensively, and ship 1.0.0.

## What changes
- **hardening** capability: nan-reductions (nansum/nanmean/nanmin/nanmax/
  nanvar/nanstd); printer refactor (element_strings split into per-kind
  helpers, cyclomatic 45 -> 10, behavior identical); zero-dep CPU-only build
  verified; comprehensive README/docs/CHANGELOG; version 1.0.0.

## Non-goals
- New numerical features beyond nan-reductions; the remaining tracked issues
  (#7/#8/#9/#11/#14) stay documented for post-1.0 work.
