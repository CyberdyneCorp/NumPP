#pragma once

#include <optional>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- binary arithmetic ufuncs ----
NUMPP_API ndarray add(const ndarray& a, const ndarray& b);
NUMPP_API ndarray subtract(const ndarray& a, const ndarray& b);
NUMPP_API ndarray multiply(const ndarray& a, const ndarray& b);
NUMPP_API ndarray divide(const ndarray& a, const ndarray& b);        // true_divide
NUMPP_API ndarray floor_divide(const ndarray& a, const ndarray& b);
NUMPP_API ndarray mod(const ndarray& a, const ndarray& b);            // remainder
NUMPP_API ndarray power(const ndarray& a, const ndarray& b);
NUMPP_API ndarray minimum(const ndarray& a, const ndarray& b);
NUMPP_API ndarray maximum(const ndarray& a, const ndarray& b);
NUMPP_API ndarray fmin(const ndarray& a, const ndarray& b);
NUMPP_API ndarray fmax(const ndarray& a, const ndarray& b);
NUMPP_API ndarray arctan2(const ndarray& a, const ndarray& b);
NUMPP_API ndarray hypot(const ndarray& a, const ndarray& b);
NUMPP_API ndarray copysign(const ndarray& a, const ndarray& b);
NUMPP_API ndarray left_shift(const ndarray& a, const ndarray& b);
NUMPP_API ndarray right_shift(const ndarray& a, const ndarray& b);

// ---- comparison (bool result) ----
NUMPP_API ndarray equal(const ndarray& a, const ndarray& b);
NUMPP_API ndarray not_equal(const ndarray& a, const ndarray& b);
NUMPP_API ndarray less(const ndarray& a, const ndarray& b);
NUMPP_API ndarray less_equal(const ndarray& a, const ndarray& b);
NUMPP_API ndarray greater(const ndarray& a, const ndarray& b);
NUMPP_API ndarray greater_equal(const ndarray& a, const ndarray& b);

// ---- logical / bitwise ----
NUMPP_API ndarray logical_and(const ndarray& a, const ndarray& b);
NUMPP_API ndarray logical_or(const ndarray& a, const ndarray& b);
NUMPP_API ndarray logical_xor(const ndarray& a, const ndarray& b);
NUMPP_API ndarray logical_not(const ndarray& a);
NUMPP_API ndarray bitwise_and(const ndarray& a, const ndarray& b);
NUMPP_API ndarray bitwise_or(const ndarray& a, const ndarray& b);
NUMPP_API ndarray bitwise_xor(const ndarray& a, const ndarray& b);
NUMPP_API ndarray invert(const ndarray& a);

// ---- unary: same-dtype ----
NUMPP_API ndarray negative(const ndarray& a);
NUMPP_API ndarray positive(const ndarray& a);
NUMPP_API ndarray absolute(const ndarray& a);
NUMPP_API ndarray sign(const ndarray& a);
NUMPP_API ndarray square(const ndarray& a);
NUMPP_API ndarray reciprocal(const ndarray& a);

// ---- unary: float-returning ----
NUMPP_API ndarray sqrt(const ndarray& a);
NUMPP_API ndarray cbrt(const ndarray& a);
NUMPP_API ndarray exp(const ndarray& a);
NUMPP_API ndarray expm1(const ndarray& a);
NUMPP_API ndarray log(const ndarray& a);
NUMPP_API ndarray log2(const ndarray& a);
NUMPP_API ndarray log10(const ndarray& a);
NUMPP_API ndarray log1p(const ndarray& a);
NUMPP_API ndarray sin(const ndarray& a);
NUMPP_API ndarray cos(const ndarray& a);
NUMPP_API ndarray tan(const ndarray& a);
NUMPP_API ndarray arcsin(const ndarray& a);
NUMPP_API ndarray arccos(const ndarray& a);
NUMPP_API ndarray arctan(const ndarray& a);
NUMPP_API ndarray sinh(const ndarray& a);
NUMPP_API ndarray cosh(const ndarray& a);
NUMPP_API ndarray tanh(const ndarray& a);
NUMPP_API ndarray arcsinh(const ndarray& a);
NUMPP_API ndarray arccosh(const ndarray& a);
NUMPP_API ndarray arctanh(const ndarray& a);
NUMPP_API ndarray deg2rad(const ndarray& a);
NUMPP_API ndarray rad2deg(const ndarray& a);
NUMPP_API ndarray floor(const ndarray& a);
NUMPP_API ndarray ceil(const ndarray& a);
NUMPP_API ndarray trunc(const ndarray& a);
NUMPP_API ndarray rint(const ndarray& a);

// ---- predicates (bool result) ----
NUMPP_API ndarray isnan(const ndarray& a);
NUMPP_API ndarray isinf(const ndarray& a);
NUMPP_API ndarray isfinite(const ndarray& a);
NUMPP_API ndarray signbit(const ndarray& a);

// ---- complex component ufuncs ----
NUMPP_API ndarray conj(const ndarray& a);          // conjugate
NUMPP_API ndarray conjugate(const ndarray& a);
NUMPP_API ndarray real(const ndarray& a);          // real part (real dtype)
NUMPP_API ndarray imag(const ndarray& a);          // imaginary part (real dtype)
NUMPP_API ndarray angle(const ndarray& a);         // phase angle (float)

// ---- out= / where= support ----
// np.copyto(dst, src, where=): cast src to dst.dtype, broadcast to dst.shape, and
// assign into dst (everywhere, or only where the mask is true). Mutates dst.
NUMPP_API void copyto(ndarray& dst, const ndarray& src, const ndarray* where = nullptr);

// out=/where= overloads of the core arithmetic ufuncs (write into and return out).
NUMPP_API ndarray add(const ndarray& a, const ndarray& b, ndarray out, const ndarray* where = nullptr);
NUMPP_API ndarray subtract(const ndarray& a, const ndarray& b, ndarray out, const ndarray* where = nullptr);
NUMPP_API ndarray multiply(const ndarray& a, const ndarray& b, ndarray out, const ndarray* where = nullptr);
NUMPP_API ndarray divide(const ndarray& a, const ndarray& b, ndarray out, const ndarray* where = nullptr);

// ---- selection ----
NUMPP_API ndarray clip(const ndarray& a, const ndarray& lo, const ndarray& hi);
NUMPP_API ndarray where(const ndarray& cond, const ndarray& a, const ndarray& b);
NUMPP_API std::vector<ndarray> nonzero(const ndarray& a);

// ---- reductions ----
NUMPP_API ndarray sum(const ndarray& a, std::optional<int64_t> axis = std::nullopt,
                      bool keepdims = false, std::optional<DType> dtype = std::nullopt);
NUMPP_API ndarray prod(const ndarray& a, std::optional<int64_t> axis = std::nullopt,
                       bool keepdims = false, std::optional<DType> dtype = std::nullopt);
NUMPP_API ndarray amin(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);
NUMPP_API ndarray amax(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);
NUMPP_API ndarray mean(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);
NUMPP_API ndarray var(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false,
                      int64_t ddof = 0);
NUMPP_API ndarray std(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false,
                      int64_t ddof = 0);
NUMPP_API ndarray any(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);
NUMPP_API ndarray all(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);

// 0-d array carrying a Python-like scalar, using NumPy NEP-50 weak promotion
// relative to `like` (int literal stays in the array's kind; float literal makes
// integer arrays float64).
NUMPP_API ndarray scalar_like(double value, DType like, bool is_float);

// ---- operators: array-array, plus weakly-promoted scalar operands ----
#define NUMPP_BINOP(OP, FUNC)                                                                       \
  inline ndarray operator OP(const ndarray& a, const ndarray& b) { return FUNC(a, b); }             \
  inline ndarray operator OP(const ndarray& a, double s) { return FUNC(a, scalar_like(s, a.dtype(), true)); } \
  inline ndarray operator OP(double s, const ndarray& b) { return FUNC(scalar_like(s, b.dtype(), true), b); } \
  inline ndarray operator OP(const ndarray& a, long long s) { return FUNC(a, scalar_like(static_cast<double>(s), a.dtype(), false)); } \
  inline ndarray operator OP(long long s, const ndarray& b) { return FUNC(scalar_like(static_cast<double>(s), b.dtype(), false), b); } \
  inline ndarray operator OP(const ndarray& a, int s) { return a OP static_cast<long long>(s); }    \
  inline ndarray operator OP(int s, const ndarray& b) { return static_cast<long long>(s) OP b; }

NUMPP_BINOP(+, add)
NUMPP_BINOP(-, subtract)
NUMPP_BINOP(*, multiply)
NUMPP_BINOP(/, divide)
NUMPP_BINOP(==, equal)
NUMPP_BINOP(!=, not_equal)
NUMPP_BINOP(<, less)
NUMPP_BINOP(<=, less_equal)
NUMPP_BINOP(>, greater)
NUMPP_BINOP(>=, greater_equal)
#undef NUMPP_BINOP

inline ndarray operator-(const ndarray& a) { return negative(a); }

}  // namespace numpp
