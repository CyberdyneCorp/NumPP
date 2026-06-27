#include <cmath>

#include "numpp/core/dtype.hpp"
#include "numpp/core/error.hpp"

namespace numpp {

DType promote_types(DType a, DType b) {
  if (a.is_extended() || b.is_extended())
    throw type_error("promote_types: only numeric dtypes are supported");
  // For two abstract dtypes (no scalar values), numpy.promote_types coincides
  // with the NEP-50 result_type already implemented.
  return result_type(a, b);
}

DType min_scalar_type(long long value) {
  if (value >= 0) return min_scalar_type(static_cast<unsigned long long>(value));
  if (value >= -128) return kInt8;
  if (value >= -32768) return kInt16;
  if (value >= -2147483648LL) return kInt32;
  return kInt64;
}

DType min_scalar_type(unsigned long long value) {
  if (value <= 255ull) return kUInt8;
  if (value <= 65535ull) return kUInt16;
  if (value <= 4294967295ull) return kUInt32;
  return kUInt64;
}

DType min_scalar_type(double value) {
  // numpy maps any float to the smallest float type whose range holds its
  // magnitude (representation need not be exact); non-finite maps to float16.
  const double av = std::fabs(value);
  if (!std::isfinite(value)) return kFloat16;
  // numpy uses a strict bound: a magnitude equal to a type's max promotes to the
  // next-wider float (e.g. 65504.0 -> float32, not float16).
  if (av < 65504.0) return kFloat16;                   // finfo(float16).max
  if (av < 3.4028234663852886e38) return kFloat32;     // finfo(float32).max
  return kFloat64;
}

}  // namespace numpp
