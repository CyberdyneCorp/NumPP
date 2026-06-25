#pragma once

#include <optional>

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

// ---- unary math ----
NUMPP_API ndarray negative(const ndarray& a);
NUMPP_API ndarray positive(const ndarray& a);
NUMPP_API ndarray absolute(const ndarray& a);
NUMPP_API ndarray sign(const ndarray& a);
NUMPP_API ndarray square(const ndarray& a);
NUMPP_API ndarray reciprocal(const ndarray& a);
NUMPP_API ndarray sqrt(const ndarray& a);
NUMPP_API ndarray exp(const ndarray& a);
NUMPP_API ndarray log(const ndarray& a);
NUMPP_API ndarray sin(const ndarray& a);
NUMPP_API ndarray cos(const ndarray& a);
NUMPP_API ndarray tan(const ndarray& a);
NUMPP_API ndarray floor(const ndarray& a);
NUMPP_API ndarray ceil(const ndarray& a);

// ---- reductions ----
NUMPP_API ndarray sum(const ndarray& a, std::optional<int64_t> axis = std::nullopt,
                      bool keepdims = false, std::optional<DType> dtype = std::nullopt);
NUMPP_API ndarray prod(const ndarray& a, std::optional<int64_t> axis = std::nullopt,
                       bool keepdims = false, std::optional<DType> dtype = std::nullopt);
NUMPP_API ndarray amin(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);
NUMPP_API ndarray amax(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);
NUMPP_API ndarray mean(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);
NUMPP_API ndarray any(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);
NUMPP_API ndarray all(const ndarray& a, std::optional<int64_t> axis = std::nullopt, bool keepdims = false);

// ---- operator overloads (delegate to ufuncs) ----
inline ndarray operator+(const ndarray& a, const ndarray& b) { return add(a, b); }
inline ndarray operator-(const ndarray& a, const ndarray& b) { return subtract(a, b); }
inline ndarray operator*(const ndarray& a, const ndarray& b) { return multiply(a, b); }
inline ndarray operator/(const ndarray& a, const ndarray& b) { return divide(a, b); }
inline ndarray operator-(const ndarray& a) { return negative(a); }

}  // namespace numpp
