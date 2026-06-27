# Bucket C — portable NumPy long-tail (no external deps)

## Why
The NumPy-parity roadmap (Tiers 1/2/A/B/C) is fully delivered in v1.1.0. What
remains vs real NumPy splits by *why* it's missing: Bucket A needs a Python
runtime/object model, Bucket B needs an external dependency — both conflict with
NumPP's clean-room, dependency-free, iOS/Android-portable charter and stay
deferred. **Bucket C is the set that is both missing and implementable in
portable, dependency-free C++**, so it is the next slice of real work.

Each feature ships as its own PR, oracle-validated against NumPy 2.1.3, merged
only after CI is green. Bugs found during the work are filed as GitHub issues and
fixed with a regression test in the same slice.

## What changes
- **error-state**: `errstate`/`seterr`/`geterr`/`seterrcall` floating-point error
  handling (warn/raise/ignore/call per division/overflow/underflow/invalid).
- **memory-overlap**: `shares_memory`/`may_share_memory` between two arrays.
- **array-iteration**: `ndindex`, `ndenumerate`, `nditer` (flat element iteration).
- **linalg** (array-API 2023): `matrix_transpose`, `vecdot`, `vector_norm`,
  `matrix_norm`, `permute_dims`.
- **datetime-completion**: `busdaycalendar` with `weekmask`/`holidays`, threaded
  through `is_busday`/`busday_count`/`busday_offset`.
- **einsum**: `optimize=` (greedy contraction-order) and `einsum_path`.
- **masked-arrays**: hard/soft masks — `harden_mask`/`soften_mask`/`hardmask`.
- **polynomial-classes**: explicit `domain`/`window` mapping and class-level `fit`.
- **string-dtype**: NumPy-2.0 `StringDType` (variable-length UTF-8 storage).

## Non-goals (deferred, with rationale)
- **Bucket A** (`object` dtype, `recarray`, `frompyfunc`/multi-arg `vectorize`,
  `ctypeslib`, `np.matrix`): need a dynamic Python object model; no clean C++
  equivalent without a runtime.
- **Bucket B** (DEFLATE `savez_compressed`, FFTW/pocketfft, `longdouble`/float128,
  Metal/Vulkan): need external dependencies / platform SDKs that break the
  no-dependency, mobile-portable goal.
