# Complete Tier B + Tier A leftovers (NumPy parity)

## Why
Tier B of the numpy-parity-roadmap — substantial new subsystems — plus the small
Tier A leftover helpers. Authored in parallel (one agent per capability) against
fixed headers and validated (oracle for exact functions, moment tests for the
non-bit-exact distributions, exact uint64 for the bitgenerator stream).

## What changes
- **random-discrete-multivariate**: geometric, negative_binomial, zipf, logseries,
  standard_t, f, wald, vonmises, noncentral_chisquare, multinomial,
  multivariate_normal, dirichlet (statistical validation)
- **bitgenerators**: Philox and SFC64 (SFC64 bit-exact with numpy; Philox
  deterministic but not yet bit-exact, tracked in #36)
- **orthopoly**: chebvander/legvander/hermvander/hermevander/lagvander and
  chebroots/legroots/hermroots/hermeroots/lagroots
- **char-strings-completion**: center/ljust/rjust/zfill/swapcase/expandtabs and
  isalpha/isdigit/isspace/isupper/islower/isalnum/istitle
- **printoptions**: format_float_positional/scientific, set/get_printoptions,
  array2string
- **stride-tricks** + **leftovers**: sliding_window_view, as_strided, piecewise,
  apply_along_axis; polyvander, polycompanion, mask_indices

## Non-goals
- Bit-exact Philox stream (#36); the orthogonal-polynomial classes with
  domain/window (only vander/roots free functions here); StringDType / split /
  encode; full array2string formatting controls; vectorize/frompyfunc — deferred.
