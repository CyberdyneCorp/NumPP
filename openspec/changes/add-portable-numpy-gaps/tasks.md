# Tasks — Bucket C (portable NumPy long-tail)

Each feature is one PR, oracle-validated, merged after CI is green. Bugs → GitHub
issue + regression test.

- [x] **error-state**: errstate/seterr/geterr/seterrcall; tests; PR + CI green
- [x] **memory-overlap**: shares_memory/may_share_memory; tests; PR + CI green
- [x] **array-iteration**: ndindex/ndenumerate/nditer; tests; PR + CI green
- [ ] **linalg array-API**: matrix_transpose/vecdot/vector_norm/matrix_norm/permute_dims; tests; PR + CI green
- [ ] **datetime**: busdaycalendar (weekmask/holidays) threaded through busday APIs; tests; PR + CI green
- [ ] **einsum**: optimize= (greedy) + einsum_path; tests; PR + CI green
- [ ] **masked hard/soft**: harden_mask/soften_mask/hardmask; tests; PR + CI green
- [ ] **polynomial domain/window+fit**: class domain/window mapping + fit; tests; PR + CI green
- [ ] **string-dtype**: NumPy-2.0 StringDType (variable-length UTF-8); tests; PR + CI green
- [ ] openspec validate --strict; docs/CHANGELOG updated; archive change
