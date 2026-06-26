# Bit-exact Philox and MT19937 BitGenerators

## Why
Philox (#36) and a standalone MT19937 (#9) were tracked as not bit-exact with
numpy — the hardest deferred items. Both are now cracked by reverse-engineering
numpy's exact algorithm from its source.

## What changes
- **random** capability:
  - **Philox** is now bit-exact with numpy.random.Philox(seed).random_raw(). Two
    numpy-specific details were the blockers: (1) numpy's M0 multiply constant is
    `0xD2E7470EE14C6C93` (numpy's own value, **not** the canonical Random123
    `0xD2B74407B1CE6E93`), and (2) numpy **increments the 256-bit counter before**
    generating each block, so the first output block is for counter 1.
  - **MT19937BitGen** added — bit-exact with numpy.random.MT19937(seed).random_raw()
    (the native 32-bit stream): state = `SeedSequence.generate_state(624)` with
    `mt[0]=0x80000000` and starting position `pos=623` (so the first draw emits
    `mt[623]` before the first twist).
  - The Philox test now asserts bit-exactness (was a determinism test).

## Non-goals
- The remaining bit-exact long-tail items: `choice(replace=False)` (#7) and the
  ziggurat normal/exponential (#8) — separate algorithms, deferred.
