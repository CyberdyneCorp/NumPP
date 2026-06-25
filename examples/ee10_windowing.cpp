// EE / DSP: window functions and spectral leakage.
//
// Taking the DFT of a finite signal implicitly multiplies it by a rectangular
// window, whose abrupt edges smear energy across the spectrum ("leakage").
// Tapered windows (Hann, Hamming, Blackman) reduce leakage by rolling the signal
// smoothly to zero at the edges, trading a wider main lobe for lower side lobes.
// The standard definitions (matching numpy.hanning/hamming/blackman, which use
// an N-1 denominator) are, for n = 0..N-1:
//   Hann     w = 0.5  - 0.5 *cos(2*pi*n/(N-1))
//   Hamming  w = 0.54 - 0.46*cos(2*pi*n/(N-1))
//   Blackman w = 0.42 - 0.5 *cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
#include "parity.hpp"
#include <cmath>
#include <cstdio>
using namespace numpp;

int main() {
  std::printf("== EE/DSP: window functions and windowed spectra ==\n");
  const int64_t N = 64;
  ndarray n = arange(0.0, static_cast<double>(N), 1.0, kFloat64);
  ndarray theta = n * (2.0 * M_PI / (N - 1));    // 2*pi*n/(N-1)

  ndarray hann = 0.5 - 0.5 * cos(theta);
  ndarray hamming = 0.54 - 0.46 * cos(theta);
  ndarray blackman = (0.42 - 0.5 * cos(theta)) + 0.08 * cos(theta * 2.0);

  ex::check("Hann window", hann, "a=np.hanning(64)");
  ex::check("Hamming window", hamming, "a=np.hamming(64)");
  ex::check("Blackman window", blackman, "a=np.blackman(64)");

  // Coherent gain of a window = mean of its samples (amplitude correction factor).
  ex::check_scalar("Hann coherent gain (mean)", mean(hann).item<double>({}),
                   ex::numpy("a=np.array([np.mean(np.hanning(64))])").item<double>({0}));

  // Apply the Hann window to a signal and take its spectrum; leakage is reduced
  // versus the raw (rectangular-windowed) FFT. Verify the windowed spectrum.
  ndarray t = linspace(0.0, 1.0, N);
  ndarray sig = sin(t * (2.0 * M_PI * 8.0));     // 8 Hz tone
  ndarray windowed = multiply(sig, hann);
  ndarray spec = fft::fft(windowed);
  ex::check("windowed signal x .* w", windowed,
            "N=64; t=np.linspace(0,1,N); sig=np.sin(2*np.pi*8*t); a=sig*np.hanning(N)");
  ex::check("FFT of windowed signal", spec,
            "N=64; t=np.linspace(0,1,N); sig=np.sin(2*np.pi*8*t); a=np.fft.fft(sig*np.hanning(N))");
  ex::check("magnitude spectrum |FFT|", absolute(spec),
            "N=64; t=np.linspace(0,1,N); sig=np.sin(2*np.pi*8*t); a=np.abs(np.fft.fft(sig*np.hanning(N)))");

  return ex::summary();
}
