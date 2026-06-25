// EE: AC steady-state analysis of a series RLC circuit.
//
// A sinusoidal source drives a resistor R, inductor L, and capacitor C in
// series. In the phasor domain (V, I are complex) the branch impedances are:
//   Z_R = R,   Z_L = j*omega*L,   Z_C = 1/(j*omega*C) = -j/(omega*C)
// so the total series impedance as a function of angular frequency omega is
//   Z(omega) = R + j*omega*L + 1/(j*omega*C) = R + jX,
//   X(omega) = omega*L - 1/(omega*C)          (net reactance).
// |Z| is the magnitude, arg(Z) the phase that the current lags/leads the
// voltage, and the current phasor for a unit source is I = V/Z with V = 1.
// At the resonant frequency omega0 = 1/sqrt(L*C) the reactance cancels, so Z
// is purely resistive (|Z| = R) and the current magnitude peaks at 1/R.
#include "parity.hpp"

#include <cmath>
#include <complex>
#include <cstdio>

using namespace numpp;

int main() {
  std::printf("== EE: Series RLC complex impedance vs frequency ==\n");

  // Component values: R = 50 ohm, L = 1 mH, C = 1 uF.
  const double R = 50.0, L = 1e-3, C = 1e-6;

  // Sweep frequency (Hz) and convert to angular frequency omega = 2*pi*f.
  ndarray f = linspace(1000.0, 10000.0, 13);   // 13 sample frequencies
  ndarray omega = f * (2.0 * M_PI);

  // Build the complex impedance array Z[i] = R + jX[i] element by element.
  const int64_t N = omega.size();
  ndarray Z = zeros({N}, kComplex128);
  for (int64_t i = 0; i < N; ++i) {
    double w = omega.item<double>({i});
    double X = w * L - 1.0 / (w * C);                 // net reactance
    Z.set_item<std::complex<double>>({i}, std::complex<double>(R, X));
  }

  // Net reactance as an array, computed with NumPP element-wise ops.
  ndarray ones_r = ones({N}, kFloat64);
  ndarray Xarr = (omega * L) - divide(ones_r, omega * C);

  // Derived quantities via NumPP ufuncs.
  ndarray magZ = absolute(Z);                          // |Z| (real array)
  ndarray phase = arctan2(imag(Z), real(Z));           // arg(Z) = atan2(X, R)
  ndarray V = full({N}, 1.0, kComplex128);             // unit source phasor
  ndarray I = divide(V, Z);                            // current phasor I = V/Z
  ndarray magI = absolute(I);                          // |I|

  // Resonance: omega0 = 1/sqrt(L*C), f0 = omega0/(2*pi).
  double omega0 = 1.0 / std::sqrt(L * C);
  double f0 = omega0 / (2.0 * M_PI);
  // |Z| exactly at resonance (reactance cancels -> should equal R).
  ndarray Zres = zeros({1}, kComplex128);
  Zres.set_item<std::complex<double>>(
      {0}, std::complex<double>(R, omega0 * L - 1.0 / (omega0 * C)));
  double magZ_res = absolute(Zres).item<double>({0});

  // Shared NumPy prelude rebuilding the same circuit and frequency grid.
  const std::string pre =
      "R=50.0; L=1e-3; C=1e-6; "
      "f=np.linspace(1000.0,10000.0,13); w=2*np.pi*f; "
      "Z=R + 1j*(w*L) + 1.0/(1j*w*C); ";

  // Verify every quantity against NumPy.
  ex::check("angular frequency omega", omega, pre + "a=w");
  ex::check("net reactance X(omega)", Xarr, pre + "a=w*L - 1.0/(w*C)");
  ex::check("complex impedance Z", Z, pre + "a=Z");
  ex::check("impedance magnitude |Z|", magZ, pre + "a=np.abs(Z)");
  ex::check("impedance phase arg(Z)", phase, pre + "a=np.angle(Z)");
  ex::check("current phasor I=V/Z", I, pre + "a=1.0/Z");
  ex::check("current magnitude |I|", magI, pre + "a=np.abs(1.0/Z)");

  // Scalars compared against NumPy-computed oracles.
  ex::check_scalar("resonant omega0 = 1/sqrt(LC)", omega0,
                   ex::numpy("a=np.array([1.0/np.sqrt(1e-3*1e-6)])").item<double>({0}));
  ex::check_scalar("resonant frequency f0 [Hz]", f0,
                   ex::numpy("a=np.array([1.0/np.sqrt(1e-3*1e-6)/(2*np.pi)])").item<double>({0}));
  ex::check_scalar("|Z| at resonance == R", magZ_res,
                   ex::numpy("a=np.array([50.0])").item<double>({0}));

  return ex::summary();
}
