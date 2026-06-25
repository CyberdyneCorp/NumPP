#include "parity.hpp"  // ex::check / ex::check_scalar / ex::summary + numpp/numpp.hpp
#include <cstdio>
using namespace numpp;

// EE concept: the Discrete Fourier Transform (DFT) decomposes a time-domain
// signal into its constituent sinusoids. Here we sample a two-tone signal,
// take its FFT, build the magnitude spectrum, map bins to physical frequencies
// with fftfreq, and recover the two tones as peaks in the positive-frequency
// half of the spectrum.
int main() {
  std::printf("== EE: Fourier Spectrum of a Two-Tone Signal ==\n");

  // ---- Sampling parameters -------------------------------------------------
  // 256 samples over a 1 s window, endpoint excluded -> sample rate fs = 256 Hz.
  const int64_t N = 256;
  ndarray t = linspace(0.0, 1.0, N, /*endpoint=*/false);  // t[k] = k/256
  const double fs = 256.0;                                 // samples per second

  // ---- Build the signal: 5 Hz tone (amp 1.0) + 12 Hz tone (amp 0.5) --------
  ndarray signal = zeros({N}, kFloat64);
  const double two_pi = 2.0 * 3.14159265358979323846;
  for (int64_t k = 0; k < N; ++k) {
    double tk = t.item<double>({k});
    double v = std::sin(two_pi * 5.0 * tk) + 0.5 * std::sin(two_pi * 12.0 * tk);
    signal.set_item<double>({k}, v);
  }

  // Sanity-check the constructed time series against NumPy.
  ex::check("signal", signal,
            "t = np.linspace(0,1,256,endpoint=False)\n"
            "a = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*12*t)");

  // ---- Forward FFT (complex spectrum) -------------------------------------
  ndarray X = fft::fft(signal);
  ex::check("fft(signal)", X,
            "t = np.linspace(0,1,256,endpoint=False)\n"
            "s = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*12*t)\n"
            "a = np.fft.fft(s)");

  // ---- Magnitude spectrum --------------------------------------------------
  // absolute() of a complex array returns a real array of magnitudes |X[k]|.
  ndarray mag = absolute(X);
  ex::check("|fft| magnitude", mag,
            "t = np.linspace(0,1,256,endpoint=False)\n"
            "s = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*12*t)\n"
            "a = np.abs(np.fft.fft(s))");

  // ---- Frequency axis: fftfreq(N, d=1/fs) gives the Hz value of each bin ----
  ndarray freqs = fft::fftfreq(N, 1.0 / fs);
  ex::check("fftfreq", freqs, "a = np.fft.fftfreq(256, d=1/256.0)");

  // ---- Identify the two dominant POSITIVE-frequency peaks ------------------
  // For a real signal the spectrum is conjugate-symmetric, so we only scan the
  // first half (bins 0..N/2-1, the non-negative frequencies). We find the
  // strongest peak, then the strongest remaining peak away from it.
  const int64_t half = N / 2;
  int64_t peak1 = 0;
  double best1 = -1.0;
  for (int64_t k = 1; k < half; ++k) {  // skip DC bin 0
    double m = mag.item<double>({k});
    if (m > best1) { best1 = m; peak1 = k; }
  }
  int64_t peak2 = 0;
  double best2 = -1.0;
  for (int64_t k = 1; k < half; ++k) {
    if (std::abs(k - peak1) < 2) continue;  // stay clear of the first peak
    double m = mag.item<double>({k});
    if (m > best2) { best2 = m; peak2 = k; }
  }

  double f1 = freqs.item<double>({peak1});  // dominant tone -> 5 Hz (amp 1.0)
  double f2 = freqs.item<double>({peak2});  // secondary tone -> 12 Hz (amp 0.5)

  // The 5 Hz tone (amplitude 1.0) must dominate the 12 Hz tone (amplitude 0.5).
  ex::check_scalar("dominant peak freq (Hz)", f1, 5.0);
  ex::check_scalar("secondary peak freq (Hz)", f2, 12.0);

  // DFT magnitude of a pure tone of amplitude A over N samples is A*N/2.
  // 5 Hz: |X| = 1.0*256/2 = 128 ; 12 Hz: |X| = 0.5*256/2 = 64.
  ex::check_scalar("dominant peak magnitude", mag.item<double>({peak1}), 128.0, 1e-4);
  ex::check_scalar("secondary peak magnitude", mag.item<double>({peak2}), 64.0, 1e-4);

  // ---- Parseval / energy check: time-domain power vs spectral power --------
  // sum(|x|^2) == (1/N) * sum(|X|^2).
  ndarray power_spec = mag * mag;
  double spectral_energy = sum(power_spec).item<double>({}) / static_cast<double>(N);
  ndarray sig_sq = signal * signal;
  double time_energy = sum(sig_sq).item<double>({});
  ex::check_scalar("Parseval energy", spectral_energy, time_energy, 1e-6);

  return ex::summary();
}
