# Add worked examples (EE math + neural network)

## Why
Demonstrate NumPP on realistic problems and prove parity beyond the unit tests:
a typical Electrical-Engineering math course plus a neural network, each
self-verifying against live NumPy.

## What changes
- **examples** capability: 7 EE-math example programs (RLC AC analysis, nodal
  analysis, Fourier spectrum, FIR convolution, state-space stability,
  least-squares fit, three-phase power) and 1 neural-network example (a 2->4->1
  MLP learning XOR with full back-prop), plus a shared `parity.hpp` harness that
  compares each result to NumPy at runtime. CMake builds all examples.

## Non-goals
- Exhaustive coverage of every EE subfield (Laplace/control/EM); the set covers
  the core course topics and the harness makes adding more trivial.
