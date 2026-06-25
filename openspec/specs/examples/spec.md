# examples Specification

## Purpose
TBD - created by archiving change add-examples. Update Purpose after archive.
## Requirements
### Requirement: Self-verifying example programs
The repository SHALL provide example programs covering core Electrical-
Engineering math (AC circuits, linear systems, Fourier/FFT, convolution/filtering,
eigen-stability, least-squares, power) and a neural network, each of which
computes its result with NumPP and verifies it against live NumPy at runtime,
exiting non-zero on any parity failure.

#### Scenario: Examples build and pass NumPy parity
- WHEN the examples are built and run
- THEN each program prints PASS for every checked quantity and exits zero

#### Scenario: Neural network learns XOR and matches NumPy
- WHEN the MLP example trains on XOR with fixed deterministic weights
- THEN its trained predictions match an identical NumPy training loop and
  converge to the XOR targets

