#include "parity.hpp"  // provides ex::check / ex::check_scalar / ex::summary and includes numpp/numpp.hpp
#include <cstdio>
#include <complex>
using namespace numpp;

// Three-phase AC power analysis.
//
// A balanced three-phase source has three voltage phasors of equal magnitude
// (here 230 V RMS) spaced 120 degrees apart: phase A at 0, phase B at -120,
// phase C at +120 degrees. Each phase drives the same load impedance
// Z = 10 + 5j ohms.
//
// Per phase we compute:
//   - phase current   I = V / Z                (Ohm's law with phasors)
//   - complex power    S = V * conj(I)         (S = P + jQ)
// and the system totals:
//   - real power       P = sum(Re(S))   [watts]
//   - reactive power   Q = sum(Im(S))   [VAR]
//   - power factor     pf = P / |S|     (cos of the load angle)
int main() {
  std::printf("== Three-Phase AC Power (phasors, S = V*conj(I)) ==\n");

  const double Vmag = 230.0;                  // RMS line-to-neutral magnitude
  const double deg2rad = std::acos(-1.0) / 180.0;
  const std::complex<double> Z(10.0, 5.0);    // load impedance per phase

  // Build the three voltage phasors V = Vmag * exp(j*theta).
  ndarray V = zeros({3}, kComplex128);
  const double angles_deg[3] = {0.0, -120.0, 120.0};
  for (int k = 0; k < 3; ++k)
    V.set_item<std::complex<double>>({k}, std::polar(Vmag, angles_deg[k] * deg2rad));

  // Impedance as a phasor array (same Z on every phase) so we can divide elem-wise.
  ndarray Zarr = zeros({3}, kComplex128);
  for (int k = 0; k < 3; ++k)
    Zarr.set_item<std::complex<double>>({k}, Z);

  // Phase currents and complex power per phase.
  ndarray I = V / Zarr;                 // I = V / Z
  ndarray S = V * conj(I);              // S = V * conj(I) = P + jQ

  // Real and reactive parts per phase (real-valued arrays).
  ndarray P = real(S);
  ndarray Q = imag(S);

  // System totals (full reductions -> 0-d arrays).
  double P_total = sum(P).item<double>({});
  double Q_total = sum(Q).item<double>({});
  double pf = P_total / std::sqrt(P_total * P_total + Q_total * Q_total);

  // NumPy oracle: rebuild the same quantities.
  const std::string setup =
      "Vmag=230.0\n"
      "ang=np.deg2rad([0.0,-120.0,120.0])\n"
      "V=Vmag*np.exp(1j*ang)\n"
      "Z=10.0+5.0j\n"
      "I=V/Z\n"
      "S=V*np.conj(I)\n";

  // Voltage phasors and the derived currents.
  ex::check("voltage phasors V", V, setup + "a=V");
  ex::check("phase currents I = V/Z", I, setup + "a=I");

  // Complex power per phase and its real/reactive split.
  ex::check("complex power S per phase", S, setup + "a=S");
  ex::check("real power P per phase", P, setup + "a=np.real(S)");
  ex::check("reactive power Q per phase", Q, setup + "a=np.imag(S)");

  // Totals and power factor.
  ndarray Ptot = ex::numpy(setup + "a=np.sum(np.real(S))");
  ndarray Qtot = ex::numpy(setup + "a=np.sum(np.imag(S))");
  ex::check_scalar("total real power P [W]", P_total, Ptot.item<double>({}));
  ex::check_scalar("total reactive power Q [VAR]", Q_total, Qtot.item<double>({}));

  // Power factor = cos(angle of Z) for a balanced load: 10/|10+5j|.
  double pf_expect = 10.0 / std::sqrt(10.0 * 10.0 + 5.0 * 5.0);
  ex::check_scalar("power factor (lagging)", pf, pf_expect);

  // |S_total| sanity: apparent power = sqrt(P^2 + Q^2).
  double S_app = std::sqrt(P_total * P_total + Q_total * Q_total);
  ndarray Sapp = ex::numpy(setup + "a=np.abs(np.sum(S))");
  ex::check_scalar("apparent power |S| [VA]", S_app, Sapp.item<double>({}));

  return ex::summary();
}
