#pragma once

#include <cstdint>

#include "numpp/core/dtype.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Machine limits for a floating/complex dtype (numpy.finfo). Complex dtypes
// report the limits of their underlying real component. All magnitudes are held
// as double (numpy promotes finfo attributes to Python floats).
struct FInfo {
  int bits;            // size of the real component in bits
  int nmant;           // explicit mantissa bits
  int nexp;            // exponent bits
  int precision;       // approximate decimal precision
  double eps;          // 2**-nmant
  double epsneg;       // 2**-(nmant+1)
  double tiny;         // smallest positive normal (== smallest_normal)
  double smallest_normal;
  double smallest_subnormal;
  double max;
  double min;          // == -max
  double resolution;   // 10**-precision
};

// Machine limits for an integer dtype (numpy.iinfo).
struct IInfo {
  int bits;
  long long min;
  unsigned long long max;
};

// numpy.finfo: floating/complex dtypes only (raises value_error otherwise).
NUMPP_API FInfo finfo(DType dt);
// numpy.iinfo: integer/bool dtypes only (raises value_error otherwise).
NUMPP_API IInfo iinfo(DType dt);

}  // namespace numpp
