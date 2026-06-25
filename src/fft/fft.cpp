#include "numpp/fft/fft.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

namespace numpp {
namespace fft {
namespace {

using cd = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;

bool is_pow2(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }
int64_t next_pow2(int64_t n) { int64_t m = 1; while (m < n) m <<= 1; return m; }

// In-place iterative radix-2 Cooley-Tukey. size must be a power of two.
// Unnormalized; `inverse` flips the exponent sign only.
void fft_pow2(std::vector<cd>& a, bool inverse) {
  const int64_t n = static_cast<int64_t>(a.size());
  for (int64_t i = 1, j = 0; i < n; ++i) {       // bit-reversal permutation
    int64_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (int64_t len = 2; len <= n; len <<= 1) {
    double ang = (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(len);
    cd wlen(std::cos(ang), std::sin(ang));
    for (int64_t i = 0; i < n; i += len) {
      cd w(1.0, 0.0);
      for (int64_t k = 0; k < len / 2; ++k) {
        cd u = a[i + k], v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
}

// Bluestein chirp-z transform: unnormalized DFT for arbitrary n.
std::vector<cd> bluestein(const std::vector<cd>& x, bool inverse) {
  const int64_t n = static_cast<int64_t>(x.size());
  const double sign = inverse ? 1.0 : -1.0;
  std::vector<cd> w(n);                          // w_j = exp(sign * i * pi * j^2 / n)
  for (int64_t j = 0; j < n; ++j) {
    double ang = sign * kPi * static_cast<double>((j * j) % (2 * n)) / static_cast<double>(n);
    w[j] = cd(std::cos(ang), std::sin(ang));
  }
  const int64_t m = next_pow2(2 * n - 1);
  std::vector<cd> a(m, cd(0)), b(m, cd(0));
  for (int64_t j = 0; j < n; ++j) a[j] = x[j] * w[j];
  b[0] = std::conj(w[0]);
  for (int64_t j = 1; j < n; ++j) { b[j] = std::conj(w[j]); b[m - j] = std::conj(w[j]); }
  fft_pow2(a, false); fft_pow2(b, false);
  for (int64_t i = 0; i < m; ++i) a[i] *= b[i];
  fft_pow2(a, true);
  for (int64_t i = 0; i < m; ++i) a[i] /= static_cast<double>(m);  // inverse-FFT normalization
  std::vector<cd> out(n);
  for (int64_t j = 0; j < n; ++j) out[j] = a[j] * w[j];
  return out;
}

// Unnormalized 1-D DFT for any length.
std::vector<cd> dft1d(std::vector<cd> x, bool inverse) {
  if (x.empty()) return x;
  if (is_pow2(static_cast<int64_t>(x.size()))) { fft_pow2(x, inverse); return x; }
  return bluestein(x, inverse);
}

DType out_dtype(DType in) {
  return (in == kFloat32 || in == kComplex64) ? kComplex64 : kComplex128;
}

// Per-norm scale factor for forward / inverse transforms of length L.
double norm_scale(const std::string& norm, int64_t L, bool inverse) {
  if (norm == "ortho") return 1.0 / std::sqrt(static_cast<double>(L));
  if (norm == "forward") return inverse ? 1.0 : 1.0 / static_cast<double>(L);
  return inverse ? 1.0 / static_cast<double>(L) : 1.0;  // backward (default)
}

ndarray transform_axis(const ndarray& a, std::optional<int64_t> n_opt, int64_t axis,
                       const std::string& norm, bool inverse) {
  const int64_t d = a.ndim();
  const int64_t ax = axis < 0 ? axis + d : axis;
  if (ax < 0 || ax >= d) throw axis_error("fft: axis out of range");
  const int64_t oldL = a.shape()[ax];
  const int64_t L = n_opt ? *n_opt : oldL;
  if (L <= 0) throw value_error("fft: invalid number of points");

  std::vector<int64_t> perm;
  for (int64_t i = 0; i < d; ++i) if (i != ax) perm.push_back(i);
  perm.push_back(ax);
  ndarray moved = a.astype(kComplex128).transpose(perm).ascontiguousarray();
  Shape mshape = moved.shape();
  Shape oshape = mshape; oshape.back() = L;
  ndarray out_moved(oshape, kComplex128, Order::C);
  const int64_t outer = oldL > 0 ? moved.size() / oldL : 0;
  const cd* src = moved.size() ? moved.typed_data<cd>() : nullptr;
  cd* dst = out_moved.size() ? out_moved.typed_data<cd>() : nullptr;
  const double scale = norm_scale(norm, L, inverse);
  for (int64_t r = 0; r < outer; ++r) {
    std::vector<cd> row(L, cd(0));
    for (int64_t j = 0; j < std::min(oldL, L); ++j) row[j] = src[r * oldL + j];
    std::vector<cd> X = dft1d(std::move(row), inverse);
    for (int64_t j = 0; j < L; ++j) dst[r * L + j] = X[j] * scale;
  }
  std::vector<int64_t> inv(d);
  for (int64_t i = 0; i < d; ++i) inv[perm[i]] = i;
  return out_moved.transpose(inv).ascontiguousarray().astype(out_dtype(a.dtype()));
}

}  // namespace

ndarray fft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  return transform_axis(a, n, axis, norm, /*inverse=*/false);
}
ndarray ifft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  return transform_axis(a, n, axis, norm, /*inverse=*/true);
}

}  // namespace fft
}  // namespace numpp
