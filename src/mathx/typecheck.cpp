#include "numpp/mathx/typecheck.hpp"

#include "numpp/core/creation.hpp"
#include "numpp/umath/ufunc.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/error.hpp"
#include <cmath>
#include <complex>
#include <limits>
#include <vector>


namespace numpp {

namespace {

// Read the single value of a (reduced) array as a double, dtype-agnostic.
inline double scalar_double(const ndarray& x) {
  ndarray c = x.astype(kFloat64).ascontiguousarray();
  return c.typed_data<double>()[0];
}

// Largest magnitude of the imaginary part of `a` (0 for a real-dtype array).
inline double max_abs_imag(const ndarray& a) {
  if (a.size() == 0) return 0.0;
  return scalar_double(amax(absolute(imag(a))));
}

// Smallest element of `a` interpreted as float64 (used to pick real vs complex).
inline double min_value(const ndarray& a) {
  if (a.size() == 0) return 0.0;
  return scalar_double(amin(a.astype(kFloat64)));
}

}  // namespace

ndarray fix(const ndarray& a) {
  return trunc(a.astype(kFloat64));
}

ndarray real_if_close(const ndarray& a, double tol) {
  if (!a.dtype().is_complex()) return a;
  const double eps = 2.220446049250313e-16;  // float64 machine epsilon
  if (max_abs_imag(a) < tol * eps) return real(a);
  return a;
}

bool iscomplexobj(const ndarray& a) { return a.dtype().is_complex(); }

bool isrealobj(const ndarray& a) { return !a.dtype().is_complex(); }

ndarray iscomplex(const ndarray& a) {
  ndarray im = imag(a);
  return not_equal(im, zeros(im.shape(), im.dtype()));
}

ndarray isreal(const ndarray& a) {
  ndarray im = imag(a);
  return equal(im, zeros(im.shape(), im.dtype()));
}

DType common_type(const std::vector<ndarray>& arrays) {
  for (const auto& arr : arrays)
    if (arr.dtype().is_complex()) return kComplex128;
  return kFloat64;  // NumPy promotes integers/reals to at least float64
}

ndarray packbits(const ndarray& a) {
  if (a.ndim() != 1) throw value_error("packbits expects a 1-D array");
  ndarray src = a.astype(kUInt8).ascontiguousarray();
  const uint8_t* in = src.typed_data<uint8_t>();
  const int64_t n = src.size();
  const int64_t nbytes = (n + 7) / 8;
  ndarray out = zeros(Shape{nbytes}, kUInt8);
  uint8_t* dst = out.typed_data<uint8_t>();
  for (int64_t i = 0; i < n; ++i)
    if (in[i]) dst[i / 8] |= static_cast<uint8_t>(1u << (7 - (i % 8)));
  return out;
}

ndarray unpackbits(const ndarray& a) {
  if (a.ndim() != 1) throw value_error("unpackbits expects a 1-D array");
  ndarray src = a.astype(kUInt8).ascontiguousarray();
  const uint8_t* in = src.typed_data<uint8_t>();
  const int64_t n = src.size();
  ndarray out = zeros(Shape{n * 8}, kUInt8);
  uint8_t* dst = out.typed_data<uint8_t>();
  for (int64_t i = 0; i < n; ++i)
    for (int b = 0; b < 8; ++b)
      dst[i * 8 + b] = static_cast<uint8_t>((in[i] >> (7 - b)) & 1u);
  return out;
}

namespace emath {

ndarray sqrt(const ndarray& a) {
  if (min_value(a) >= 0.0) return numpp::sqrt(a.astype(kFloat64));
  return numpp::sqrt(a.astype(kComplex128));
}

ndarray log(const ndarray& a) {
  if (min_value(a) > 0.0) return numpp::log(a.astype(kFloat64));
  return numpp::log(a.astype(kComplex128));
}

ndarray power(const ndarray& a, const ndarray& b) {
  if (min_value(a) < 0.0)
    return numpp::power(a.astype(kComplex128), b.astype(kComplex128));
  return numpp::power(a.astype(kFloat64), b.astype(kFloat64));
}

}  // namespace emath

}  // namespace numpp
