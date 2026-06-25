# Add random (Phase 6)

## Why

Phase 6 adds `numpy.random` parity. The headline requirement is **bit-exact
reproducibility**: a NumPP `Generator(seed)` must produce the same stream as
`numpy.random.Generator(np.random.PCG64(seed))`, so seeded code is portable.

## What changes

- **random** capability: numpy's `SeedSequence` entropy mixer, the `PCG64`
  BitGenerator (XSL-RR 128/64) and `MT19937`, a `Generator` class
  (`random`/`integers`/`uniform`/`normal`/`standard_normal`/`exponential`/
  `poisson`/`binomial`/`choice`/`shuffle`/`permutation`), and the legacy
  `RandomState` API.

## Reuse vs rewrite

- Clean-room reimplementation of numpy's exact algorithms (SeedSequence hashmix,
  PCG64 XSL-RR, 53-bit double method, Lemire bounded integers, ziggurat normal)
  so the bit stream matches. 128-bit state via `unsigned __int128` (clang/gcc).

## Non-goals

- Philox/SFC64 BitGenerators (can be added later).
- Bit-exact parity for distributions whose numpy implementation is not
  reproducible in portable C++ — those get statistical (KS/moment) validation and
  a tracking issue instead.
