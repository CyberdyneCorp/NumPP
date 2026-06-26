#pragma once

#include <optional>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace ma {

// Minimal masked-array support (numpy.ma). The mask is a bool array where true
// marks a masked (ignored) element.
struct MaskedArray {
  ndarray data;
  ndarray mask;
};

NUMPP_API MaskedArray masked_array(const ndarray& data, const ndarray& mask);
NUMPP_API MaskedArray masked_where(const ndarray& condition, const ndarray& a);  // mask where condition true
NUMPP_API MaskedArray masked_invalid(const ndarray& a);                          // mask NaN / inf
NUMPP_API MaskedArray masked_equal(const ndarray& a, double value);
NUMPP_API MaskedArray masked_greater(const ndarray& a, double value);

NUMPP_API ndarray filled(const MaskedArray& m, double fill_value);  // masked -> fill_value
NUMPP_API ndarray compressed(const MaskedArray& m);                 // 1-D of unmasked values
NUMPP_API int64_t count(const MaskedArray& m);                      // number of unmasked

// Reductions over the unmasked elements (flattened), returning a 0-d array.
NUMPP_API ndarray sum(const MaskedArray& m);
NUMPP_API ndarray mean(const MaskedArray& m);
NUMPP_API ndarray min(const MaskedArray& m);
NUMPP_API ndarray max(const MaskedArray& m);

}  // namespace ma
}  // namespace numpp
