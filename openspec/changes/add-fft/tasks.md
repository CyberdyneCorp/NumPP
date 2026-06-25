# Tasks — Phase 5 fft

- [x] 1 DFT engine: radix-2 iterative Cooley-Tukey + Bluestein (arbitrary/prime n)
- [x] 2 fft/ifft (axis, n, norm in {backward,ortho,forward}); complex128/complex64 output
- [x] 3 rfft/irfft (+ n), hfft/ihfft
- [x] 4 fftfreq/rfftfreq/fftshift/ifftshift
- [x] 5 fft2/ifft2/rfft2/irfft2, fftn/ifftn/rfftn/irfftn (axes/s)
- [x] 6 NumPy-oracle tests for every routine; issue + regression per divergence
- [x] 7 openspec validate --strict; clang+gcc + ASan/UBSan; PR + merge + archive
