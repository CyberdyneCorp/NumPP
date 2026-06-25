# Add FFT (Phase 5)

## Why

Phase 5 adds `numpy.fft` parity: discrete Fourier transforms in pure C++ that
work on any length and on any axis, so signal/spectral code runs on NumPP.

## What changes

- **fft** capability: a complex 1-D DFT engine — iterative radix-2 Cooley-Tukey
  for power-of-two lengths and a Bluestein (chirp-z) transform for arbitrary or
  prime lengths (correct for all n) — exposed as `fft`/`ifft`, the real
  transforms `rfft`/`irfft`/`hfft`/`ihfft`, the 2-D and N-D variants
  (`fft2`/`ifft2`/`rfft2`/`irfft2`, `fftn`/`ifftn`/`rfftn`/`irfftn`), and the
  helpers `fftfreq`/`rfftfreq`/`fftshift`/`ifftshift`.
- `norm` in {`backward` (default), `ortho`, `forward`}; `n`/`s`/`axis`/`axes`
  parameters with NumPy truncate/zero-pad semantics.

## Reuse vs rewrite

- Pure-C++ DFT (no FFTW/pocketfft dependency) so the mobile build needs nothing.
  Output is computed in double precision and cast to complex64/complex128 to
  match NumPy's input-precision-preserving dtype rule.

## Non-goals

- FFTW/pocketfft backends (a future optional acceleration, like BLAS/LAPACK).
- Long-double transforms.
