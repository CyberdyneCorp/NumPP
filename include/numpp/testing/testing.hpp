#pragma once

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace testing {

// numpy.testing-style helpers. The assert_* functions throw value_error with a
// descriptive message on mismatch and return normally on success.
NUMPP_API bool array_equal(const ndarray& a, const ndarray& b);    // numpy.array_equal
NUMPP_API bool array_equiv(const ndarray& a, const ndarray& b);    // numpy.array_equiv (broadcast)

NUMPP_API void assert_array_equal(const ndarray& a, const ndarray& b);
NUMPP_API void assert_allclose(const ndarray& a, const ndarray& b, double rtol = 1e-7, double atol = 0.0);
NUMPP_API void assert_array_almost_equal(const ndarray& a, const ndarray& b, int decimal = 6);
NUMPP_API void assert_array_less(const ndarray& a, const ndarray& b);

}  // namespace testing
}  // namespace numpp
