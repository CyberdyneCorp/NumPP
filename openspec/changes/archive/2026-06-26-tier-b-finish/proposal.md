# Finish Tier B (NumPy parity)

## Why
Closes the remaining functional gaps in Tier B of the numpy-parity-roadmap: the
orthogonal-polynomial calculus + classes, and the leftover char / random / stride
sub-items. Authored in parallel (one agent per capability) against fixed headers.

## What changes
- **polynomial-classes**: chebder/chebint, legder/legint, hermder/hermint,
  hermeder/hermeint, lagder/lagint; and the Chebyshev/Legendre/Hermite/HermiteE/
  Laguerre classes (eval/deriv/integ/roots, default identity domain)
- **char-strings-completion**: join, encode, decode, partition
- **random-discrete-multivariate**: hypergeometric, noncentral_f, bytes
- **stride-tricks**: vectorize, apply_over_axes

## Non-goals
- Domain/window mapping and `fit()` on the polynomial classes; `*fromroots`/
  `*gauss`; numpy-bit-exact Philox (#36); char `split`/`StringDType`;
  `frompyfunc` — deferred.
