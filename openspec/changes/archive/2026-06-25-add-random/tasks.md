# Tasks — Phase 6 random

- [x] 1 SeedSequence (exact hashmix) + PCG64 (XSL-RR 128/64); bit-exact raw stream + random doubles
- [x] 2 Generator: integers (Lemire bounded), uniform, shuffle/permutation/choice — exact parity
- [x] 3 Distributions: standard_normal (ziggurat), normal, exponential, poisson, binomial, gamma, beta, chisquare (bit-exact where feasible, else statistical + issue)
- [x] 4 MT19937 BitGenerator + legacy RandomState (rand/randn/randint/random_sample/normal) exact parity
- [x] 5 NumPy-oracle tests for every routine; issue + regression per divergence
- [x] 6 openspec validate --strict; clang+gcc + ASan/UBSan; PR + merge + archive
