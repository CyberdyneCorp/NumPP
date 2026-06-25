#include "numpp/fft/fft.hpp"

#include "numpp/core/creation.hpp"
#include "numpp/umath/ufunc.hpp"

#include <algorithm>
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

namespace {
int64_t norm_axis(int64_t axis, int64_t d) { return axis < 0 ? axis + d : axis; }
std::string swap_norm(const std::string& norm) {
  if (norm == "backward") return "forward";
  if (norm == "forward") return "backward";
  return norm;  // ortho
}
ndarray slice_axis(const ndarray& a, int64_t axis, int64_t start, int64_t stop) {
  std::vector<IndexItem> items;
  for (int64_t i = 0; i < a.ndim(); ++i)
    items.push_back(i == axis ? IndexItem{Slice{start, stop, 1}} : IndexItem{Slice{}});
  return a.index(items);
}
DType real_dtype(DType in) { return (in == kComplex64 || in == kFloat32) ? kFloat32 : kFloat64; }

// Rebuild a full length-`nout` complex spectrum from a half spectrum via Hermitian symmetry.
ndarray build_hermitian(const ndarray& a, int64_t ax, int64_t nout) {
  const int64_t d = a.ndim(), m = a.shape()[ax];
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < d; ++i) if (i != ax) perm.push_back(i);
  perm.push_back(ax);
  ndarray moved = a.astype(kComplex128).transpose(perm).ascontiguousarray();
  Shape oshape = moved.shape(); oshape.back() = nout;
  ndarray out(oshape, kComplex128, Order::C);
  const int64_t outer = m > 0 ? moved.size() / m : 0;
  const cd* src = moved.size() ? moved.typed_data<cd>() : nullptr;
  cd* dst = out.size() ? out.typed_data<cd>() : nullptr;
  for (int64_t r = 0; r < outer; ++r) {
    auto row = [&](int64_t k) { return k < m ? src[r * m + k] : cd(0); };
    dst[r * nout + 0] = row(0);
    for (int64_t k = 1; 2 * k < nout; ++k) { cd s = row(k); dst[r * nout + k] = s; dst[r * nout + (nout - k)] = std::conj(s); }
    if (nout % 2 == 0) dst[r * nout + nout / 2] = row(nout / 2);
  }
  std::vector<int64_t> inv(d);
  for (int64_t i = 0; i < d; ++i) inv[perm[i]] = i;
  return out.transpose(inv).ascontiguousarray();
}

ndarray roll_axis(const ndarray& a, int64_t shift, int64_t axis) {
  const int64_t d = a.ndim(), ax = norm_axis(axis, d), n = a.shape()[ax];
  if (n == 0) return a.copy();
  const int64_t sh = ((shift % n) + n) % n, es = a.itemsize();
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < d; ++i) if (i != ax) perm.push_back(i);
  perm.push_back(ax);
  ndarray moved = a.transpose(perm).ascontiguousarray();
  ndarray out(moved.shape(), a.dtype(), Order::C);
  const int64_t outer = moved.size() / n;
  const char* src = moved.bytes(); char* dst = out.bytes();
  for (int64_t r = 0; r < outer; ++r)
    for (int64_t j = 0; j < n; ++j)
      std::memcpy(dst + (r * n + (j + sh) % n) * es, src + (r * n + j) * es, es);
  std::vector<int64_t> inv(d);
  for (int64_t i = 0; i < d; ++i) inv[perm[i]] = i;
  return out.transpose(inv).ascontiguousarray();
}
}  // namespace

ndarray rfft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  const int64_t ax = norm_axis(axis, a.ndim());
  const int64_t L = n ? *n : a.shape()[ax];
  return slice_axis(fft(a, n, axis, norm), ax, 0, L / 2 + 1).ascontiguousarray();
}
ndarray irfft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  const int64_t ax = norm_axis(axis, a.ndim());
  const int64_t nout = n ? *n : 2 * (a.shape()[ax] - 1);
  ndarray full = build_hermitian(a, ax, nout);
  return real(ifft(full, nout, axis, norm)).astype(real_dtype(a.dtype()));
}
ndarray hfft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  const int64_t ax = norm_axis(axis, a.ndim());
  const int64_t nout = n ? *n : 2 * (a.shape()[ax] - 1);
  return irfft(conj(a), nout, axis, swap_norm(norm));
}
ndarray ihfft(const ndarray& a, std::optional<int64_t> n, int64_t axis, const std::string& norm) {
  return conj(rfft(a, n, axis, swap_norm(norm)));
}

namespace {
std::vector<int64_t> default_axes(const ndarray& a, std::optional<std::vector<int64_t>> axes,
                                  bool two_d) {
  if (axes) return *axes;
  std::vector<int64_t> ax;
  if (two_d) { ax = {-2, -1}; }
  else for (int64_t i = 0; i < a.ndim(); ++i) ax.push_back(i);
  return ax;
}
std::optional<int64_t> s_at(const std::optional<std::vector<int64_t>>& s, size_t i) {
  if (s && i < s->size()) return (*s)[i];
  return std::nullopt;
}
}  // namespace

ndarray fftn(const ndarray& a, std::optional<std::vector<int64_t>> s,
             std::optional<std::vector<int64_t>> axes, const std::string& norm) {
  std::vector<int64_t> ax = default_axes(a, axes, false);
  ndarray r = a;
  for (size_t i = 0; i < ax.size(); ++i) r = fft(r, s_at(s, i), ax[i], norm);
  return r;
}
ndarray ifftn(const ndarray& a, std::optional<std::vector<int64_t>> s,
              std::optional<std::vector<int64_t>> axes, const std::string& norm) {
  std::vector<int64_t> ax = default_axes(a, axes, false);
  ndarray r = a;
  for (size_t i = 0; i < ax.size(); ++i) r = ifft(r, s_at(s, i), ax[i], norm);
  return r;
}
ndarray fft2(const ndarray& a, std::optional<std::vector<int64_t>> s,
             std::optional<std::vector<int64_t>> axes, const std::string& norm) {
  return fftn(a, s, default_axes(a, axes, true), norm);
}
ndarray ifft2(const ndarray& a, std::optional<std::vector<int64_t>> s,
              std::optional<std::vector<int64_t>> axes, const std::string& norm) {
  return ifftn(a, s, default_axes(a, axes, true), norm);
}
ndarray rfftn(const ndarray& a, std::optional<std::vector<int64_t>> s,
              std::optional<std::vector<int64_t>> axes, const std::string& norm) {
  std::vector<int64_t> ax = default_axes(a, axes, false);
  ndarray r = rfft(a, s_at(s, ax.size() - 1), ax.back(), norm);   // real transform on last axis
  for (size_t i = 0; i + 1 < ax.size(); ++i) r = fft(r, s_at(s, i), ax[i], norm);
  return r;
}
ndarray irfftn(const ndarray& a, std::optional<std::vector<int64_t>> s,
               std::optional<std::vector<int64_t>> axes, const std::string& norm) {
  std::vector<int64_t> ax = default_axes(a, axes, false);
  ndarray r = a;
  for (size_t i = 0; i + 1 < ax.size(); ++i) r = ifft(r, s_at(s, i), ax[i], norm);
  return irfft(r, s_at(s, ax.size() - 1), ax.back(), norm);       // real inverse on last axis
}
ndarray rfft2(const ndarray& a, std::optional<std::vector<int64_t>> s,
              std::optional<std::vector<int64_t>> axes, const std::string& norm) {
  return rfftn(a, s, default_axes(a, axes, true), norm);
}
ndarray irfft2(const ndarray& a, std::optional<std::vector<int64_t>> s,
               std::optional<std::vector<int64_t>> axes, const std::string& norm) {
  return irfftn(a, s, default_axes(a, axes, true), norm);
}

ndarray fftfreq(int64_t n, double d) {
  ndarray f(Shape{n}, kFloat64, Order::C);
  const double inv = 1.0 / (static_cast<double>(n) * d);
  const int64_t half = (n - 1) / 2 + 1;
  for (int64_t i = 0; i < n; ++i) f.set_item<double>({i}, (i < half ? i : i - n) * inv);
  return f;
}
ndarray rfftfreq(int64_t n, double d) {
  const int64_t m = n / 2 + 1;
  ndarray f(Shape{m}, kFloat64, Order::C);
  const double inv = 1.0 / (static_cast<double>(n) * d);
  for (int64_t i = 0; i < m; ++i) f.set_item<double>({i}, i * inv);
  return f;
}
ndarray fftshift(const ndarray& a, std::optional<std::vector<int64_t>> axes) {
  std::vector<int64_t> ax;
  if (axes) ax = *axes; else for (int64_t i = 0; i < a.ndim(); ++i) ax.push_back(i);
  ndarray r = a;
  for (int64_t x : ax) { int64_t n = a.shape()[norm_axis(x, a.ndim())]; r = roll_axis(r, n / 2, x); }
  return r;
}
ndarray ifftshift(const ndarray& a, std::optional<std::vector<int64_t>> axes) {
  std::vector<int64_t> ax;
  if (axes) ax = *axes; else for (int64_t i = 0; i < a.ndim(); ++i) ax.push_back(i);
  ndarray r = a;
  for (int64_t x : ax) { int64_t n = a.shape()[norm_axis(x, a.ndim())]; r = roll_axis(r, -(n / 2), x); }
  return r;
}

}  // namespace fft
}  // namespace numpp
