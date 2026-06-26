# NumPy parity roadmap (gap backlog)

## Why
NumPP v1.0.0 covers the core NumPy surface (ndarray, dtypes incl. strings/
datetime/structured, ufuncs, reductions, sorting, linalg, fft, random, I/O).
This change records — as a living backlog — the NumPy features that are still
missing, prioritized into tiers, so future work flows through OpenSpec.

The full per-module gap analysis lives in `docs/numpy-parity-gaps.md`. This
change is intentionally **not implemented/archived**; it is the tracking artifact.
Each tier item below should graduate into its own OpenSpec change when picked up.

## What changes
- **numpy-parity** capability: a documented set of target requirements for the
  missing NumPy areas (array manipulation, statistics, grids/indexing, ufunc
  extras, polynomials, einsum, full random distributions, text I/O, char ops),
  plus the suggested implementation order.

## Tiers (summary)
**Status (2026-06-26): the original Tier 1 + Tier 2 backlog is fully delivered** —
12 capabilities merged + archived (array-manipulation, statistics, creation-grids,
advanced-indexing, ufuncs-extras, signal-poly, sorting-extras, einsum,
random-distributions, polynomial-package, text-io, char-strings). Baseline now
holds 28 capabilities; the suite runs 421 cases / 1246 oracle checks, 0
divergences (one bug found & fixed, #26). This roadmap is refreshed to track
**what remains** — the advanced options deferred from those changes (their
"Non-goals") plus the original Tier 3 — regrouped into three new tiers.

- **Tier A — complete partially-done modules** (small, high value): array
  constructors from data (array/asarray/fromiter/frombuffer/mgrid/ogrid),
  manipulation extras (block/dsplit/trim_zeros/extra pad modes), indexing
  completion (fancy integer + boolean ndarray subscripting, put_along_axis, ix_,
  fill_diagonal, *_indices), statistics extras (histogram2d/histogramdd/
  nanquantile/weighted cov/N-D gradient), ufunc & type-check completion
  (fix/real_if_close/iscomplexobj/emath/packbits), poly1d class.
- **Tier B — substantial new subsystems**: discrete & multivariate random
  (geometric/hypergeometric/multinomial/multivariate_normal/dirichlet/standard_t/
  f/...), BitGenerators (Philox/SFC64/standalone MT19937), the orthogonal-
  polynomial classes (Chebyshev/Legendre/Hermite/HermiteE/Laguerre), char/string
  completion (split/join/encode/center/predicates/StringDType), print options &
  float formatting, stride tricks & functional helpers (sliding_window_view/
  as_strided/vectorize/piecewise/apply_along_axis).
- **Tier C — large / specialized**: masked arrays (numpy.ma), object dtype /
  recarray / numpy.testing, datetime business-day + unit conversion + structured
  .npy (#14), real GPU backends behind the weak vtables, the bit-exact long-tail
  (#7/#8/#9), and interop/misc (memmap, DLPack/array-API, ctypeslib, FFTW).

## Non-goals
- Implementing any tier here; this change is the roadmap. Each unchecked item in
  `tasks.md` graduates into its own OpenSpec change when started.
