// EE: Control systems — Bode plot of a second-order system.
//
// The frequency response of a transfer function H(s) is found by evaluating it
// on the imaginary axis, s = j*omega. A Bode plot shows:
//   - magnitude in decibels:  20*log10(|H(jw)|)
//   - phase in degrees:       angle(H(jw)) * 180/pi
// over a logarithmically-spaced frequency sweep. Here H is a standard 2nd-order
// low-pass system  H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)  with natural
// frequency wn and damping ratio zeta — the textbook resonant system whose Bode
// magnitude peaks near wn for light damping and rolls off at -40 dB/decade.
#include "parity.hpp"
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
using namespace numpp;

static ndarray polyval(const std::vector<double>& c, const ndarray& s) {
  ndarray r = full(s.shape(), c[0], kComplex128);
  for (size_t k = 1; k < c.size(); ++k) r = r * s + c[k];
  return r;
}

int main() {
  std::printf("== EE: Bode plot of H(s) = wn^2/(s^2 + 2*zeta*wn*s + wn^2) ==\n");
  const double wn = 10.0, zeta = 0.2;
  std::vector<double> num = {wn * wn};
  std::vector<double> den = {1.0, 2.0 * zeta * wn, wn * wn};

  // Log-spaced frequency sweep: omega = 10^linspace(-1, 2, 60)  (== np.logspace).
  ndarray omega = exp(linspace(-1.0, 2.0, 60) * std::log(10.0));
  ex::check("logspace frequency grid", omega, "a=np.logspace(-1,2,60)");

  // s = j*omega, then H(jw).
  ndarray s = zeros({omega.size()}, kComplex128);
  for (int64_t i = 0; i < omega.size(); ++i) s.set_item<std::complex<double>>({i}, std::complex<double>(0.0, omega.item<double>({i})));
  ndarray H = divide(polyval(num, s), polyval(den, s));

  ndarray mag_db = multiply(log10(absolute(H)), full(H.shape(), 20.0, kFloat64));
  ndarray phase_deg = angle(H) * (180.0 / M_PI);

  const std::string pre = "wn=10.0; zeta=0.2; w=np.logspace(-1,2,60); s=1j*w; "
                          "H=(wn**2)/(s**2 + 2*zeta*wn*s + wn**2); ";
  ex::check("frequency response H(jw)", H, pre + "a=H");
  ex::check("magnitude (dB) = 20*log10|H|", mag_db, pre + "a=20*np.log10(np.abs(H))");
  ex::check("phase (degrees)", phase_deg, pre + "a=np.angle(H, deg=True)");

  // Peak resonance magnitude and the -3 dB bandwidth-ish readouts.
  double peak_db = amax(mag_db).item<double>({});
  ex::check_scalar("peak magnitude [dB]", peak_db,
                   ex::numpy(pre + "a=np.array([np.max(20*np.log10(np.abs(H)))])").item<double>({0}), 1e-4);

  return ex::summary();
}
