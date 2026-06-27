# Tasks — Complete the NumPy foundation

Each item is one oracle-validated PR (oracle vs NumPy 2.1.3), merged after CI green.
Bugs → GitHub issue + regression test.

## Tier 1 — pervasive, cheap, pure numpy.*
- [ ] **machine-limits**: `finfo`/`iinfo` (eps, tiny, max, min, resolution, bits, nmant/nexp); tests; PR + CI green
- [ ] **inf/close predicates**: `isclose`, `isposinf`, `isneginf`; tests; PR + CI green
- [ ] **trapz**: `trapz`/`trapezoid` (x= and dx=, axis); tests; PR + CI green
- [ ] **dtype casting**: `promote_types`, `min_scalar_type`; tests; PR + CI green

## Tier 2 — completeness of existing features
- [ ] **batched linalg**: stacked last-two-axes for solve/inv/det/slogdet/matrix_power/
      cholesky/qr/eig/eigh/eigvals/eigvalsh/svd/svdvals/pinv/matrix_rank/lstsq; tests; PR + CI green
- [ ] **tensorsolve/tensorinv**: add the two missing numpy.linalg functions; tests; PR + CI green
- [ ] **einsum ellipsis**: `...` broadcasting subscripts; tests; PR + CI green
- [ ] **interp options**: `left`/`right`/`period` on `interp`; tests; PR + CI green

## Wrap-up
- [ ] openspec validate --strict; docs/CHANGELOG updated; archive change
