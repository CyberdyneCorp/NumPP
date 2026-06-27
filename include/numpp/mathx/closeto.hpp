#pragma once

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Elementwise closeness test, the array-returning form of allclose.
// abs(a - b) <= atol + rtol * abs(b), with NumPy broadcasting. Matching
// infinities compare close; opposite/differing infinities do not. When
// equal_nan is true, NaN positions in both inputs compare close.
NUMPP_API ndarray isclose(const ndarray& a, const ndarray& b, double rtol = 1e-5,
                          double atol = 1e-8, bool equal_nan = false);

// Bool array: true where the element is positive infinity.
NUMPP_API ndarray isposinf(const ndarray& a);

// Bool array: true where the element is negative infinity.
NUMPP_API ndarray isneginf(const ndarray& a);

}  // namespace numpp
