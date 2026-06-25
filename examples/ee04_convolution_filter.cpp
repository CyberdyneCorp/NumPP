#include "parity.hpp"  // ex::check / ex::check_scalar / ex::summary (+ numpp/numpp.hpp)
#include <cstdio>
using namespace numpp;

// EE concept: a FIR (finite impulse response) filter computes y[n] = sum_k h[k] x[n-k],
// i.e. the LINEAR CONVOLUTION of the input signal x with the impulse response h.
// Here h is a 3-tap moving-average kernel (a simple low-pass smoother).
//
// The CONVOLUTION THEOREM says convolution in time == multiplication in frequency:
//     conv(x, h)  <==>  IFFT( FFT(x) . FFT(h) )
// To get the *full* linear convolution (not circular), we zero-pad both signals to
// length nx + nh - 1 before the FFT, so circular wrap-around cannot corrupt any tap.
int main() {
  std::printf("== EE04: FIR Filtering via the Convolution Theorem (FFT) ==\n");

  const int nx = 20, nh = 3;
  const int N = nx + nh - 1;  // length of the full linear convolution = 22

  // Deterministic test signal: a low-frequency tone plus a slower component.
  // (Same closed form is reproduced in the NumPy oracle below.)
  ndarray idx = arange(0.0, double(nx), 1.0, kFloat64);
  ndarray x = sin(idx * 0.3) + cos(idx * 0.1) * 0.5;

  // 3-tap moving-average kernel: h = [1/3, 1/3, 1/3]; its taps sum to 1 (unity DC gain).
  ndarray h = full({nh}, 1.0 / 3.0, kFloat64);

  // --- Convolution via FFT -------------------------------------------------
  // fft(a, N) zero-pads a to length N, then transforms. Multiply the spectra,
  // invert, and take the real part (the imaginary part is round-off noise).
  ndarray X = fft::fft(x, N);
  ndarray H = fft::fft(h, N);
  ndarray Y = X * H;                 // element-wise complex product == convolution
  ndarray y_complex = fft::ifft(Y);  // already length N, no extra padding
  ndarray y = real(y_complex);       // the FIR output, length N

  // The headline check: FFT-based result must equal numpy.convolve(x, h, 'full').
  // Shared NumPy preamble that rebuilds x and h exactly.
  const std::string pre =
      "i = np.arange(20.0); "
      "x = np.sin(0.3*i) + 0.5*np.cos(0.1*i); "
      "h = np.full(3, 1.0/3.0); ";

  ex::check("full linear convolution (fft == np.convolve)", y,
            pre + "a = np.convolve(x, h)");

  // Sanity: the kernel sums to 1 (DC gain of a normalized moving average).
  ex::check_scalar("kernel taps sum to 1", sum(h).item<double>({}), 1.0);

  // The IFFT output should be (numerically) real: imaginary energy ~ 0.
  double max_imag = amax(absolute(imag(y_complex))).item<double>({});
  ex::check_scalar("ifft output is real (max |Im| ~ 0)", max_imag, 0.0, 1e-9);

  // Output length of a full convolution is nx + nh - 1.
  ex::check_scalar("output length == nx + nh - 1", double(y.size()), double(N));

  // Boundary tap: first sample only sees x[0] through h[0]  ->  y[0] = h[0]*x[0].
  double y0_expected = (1.0 / 3.0) * x.item<double>({0});
  ex::check_scalar("leading edge y[0] = h[0]*x[0]", y.item<double>({0}), y0_expected);

  // Interior samples of a length-3 averager equal the mean of 3 consecutive inputs.
  // y[k] = (x[k] + x[k-1] + x[k-2]) / 3 for 2 <= k <= nx-1.  Check one interior point.
  int k = 10;
  double interior = (x.item<double>({k}) + x.item<double>({k - 1}) +
                     x.item<double>({k - 2})) / 3.0;
  ex::check_scalar("interior y[10] = mean of 3 taps", y.item<double>({k}), interior);

  // Cross-check the spectra route against NumPy's own FFT pipeline (same algorithm).
  ex::check("spectrum product X*H matches numpy", Y,
            pre + "a = np.fft.fft(x, 22) * np.fft.fft(h, 22)");

  return ex::summary();
}
