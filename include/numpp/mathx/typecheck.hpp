#pragma once

#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- rounding / cleanup ----
NUMPP_API ndarray fix(const ndarray& a);                              // round toward zero (float result)
NUMPP_API ndarray real_if_close(const ndarray& a, double tol = 100);  // drop tiny imaginary parts

// ---- type inspection ----
NUMPP_API bool iscomplexobj(const ndarray& a);
NUMPP_API bool isrealobj(const ndarray& a);
NUMPP_API ndarray iscomplex(const ndarray& a);   // bool array: nonzero imaginary part
NUMPP_API ndarray isreal(const ndarray& a);      // bool array: zero imaginary part
NUMPP_API DType common_type(const std::vector<ndarray>& arrays);  // numpy.common_type (>= float)

// ---- bit packing ----
NUMPP_API ndarray packbits(const ndarray& a);    // 1-D bool/uint8 -> packed uint8
NUMPP_API ndarray unpackbits(const ndarray& a);  // 1-D uint8 -> bits (uint8 0/1)

// ---- complex-promoting math (numpy.emath / scimath) ----
namespace emath {
NUMPP_API ndarray sqrt(const ndarray& a);   // negative -> complex
NUMPP_API ndarray log(const ndarray& a);    // <=0 -> complex
NUMPP_API ndarray power(const ndarray& a, const ndarray& b);
}  // namespace emath

}  // namespace numpp
