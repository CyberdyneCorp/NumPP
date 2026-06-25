# fft Specification

## Purpose
TBD - created by archiving change add-fft. Update Purpose after archive.
## Requirements
### Requirement: 1-D forward and inverse FFT

NumPP SHALL provide `fft` and `ifft` computing the 1-D discrete Fourier transform
and its inverse along a chosen axis, for any length `n` (power-of-two via
Cooley-Tukey, arbitrary/prime via Bluestein), matching `numpy.fft`. The `n`
parameter SHALL truncate or zero-pad the input along the axis. The `norm`
parameter SHALL support `backward`, `ortho`, and `forward`.

#### Scenario: Round trip
- GIVEN a 1-D array x
- WHEN `ifft(fft(x))` is computed
- THEN the result equals x within tolerance

#### Scenario: Prime length matches NumPy
- WHEN `fft(x)` is computed for an array of prime length
- THEN it equals `numpy.fft.fft(x)` within tolerance

#### Scenario: Padding and truncation
- WHEN `fft(x, n)` is computed with n larger or smaller than the axis length
- THEN the input is zero-padded or truncated to n and the result equals `numpy.fft.fft(x, n)`

#### Scenario: Orthonormal norm
- WHEN `fft(x, norm="ortho")` is computed
- THEN it equals `numpy.fft.fft(x, norm="ortho")` within tolerance

### Requirement: Real-input transforms

NumPP SHALL provide `rfft`/`irfft` (and `hfft`/`ihfft`) matching `numpy.fft`,
where `rfft` returns the non-redundant half spectrum of length `n//2 + 1`.

#### Scenario: rfft length and values
- WHEN `rfft(x)` is computed for a real array of length n
- THEN the result has length `n//2 + 1` and equals `numpy.fft.rfft(x)`

#### Scenario: irfft inverts rfft
- WHEN `irfft(rfft(x), n)` is computed
- THEN the result equals the original real array within tolerance

### Requirement: 2-D and N-D transforms

NumPP SHALL provide `fft2`/`ifft2`/`rfft2`/`irfft2` and
`fftn`/`ifftn`/`rfftn`/`irfftn`, applying the 1-D transform along each requested
axis, matching `numpy.fft`.

#### Scenario: 2-D transform
- WHEN `fft2(x)` is computed for a 2-D array
- THEN it equals `numpy.fft.fft2(x)` within tolerance

### Requirement: Frequency helpers

NumPP SHALL provide `fftfreq`, `rfftfreq`, `fftshift`, and `ifftshift` matching
`numpy.fft`.

#### Scenario: fftfreq
- WHEN `fftfreq(n, d)` is computed
- THEN it equals `numpy.fft.fftfreq(n, d)`

#### Scenario: fftshift round trip
- WHEN `ifftshift(fftshift(x))` is computed
- THEN the result equals x

