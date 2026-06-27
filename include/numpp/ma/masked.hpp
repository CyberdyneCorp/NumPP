#pragma once

#include <optional>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace ma {

// Minimal masked-array support (numpy.ma). The mask is a bool array where true
// marks a masked (ignored) element. `hard` selects hard vs soft mask semantics
// for assignment (numpy.ma defaults to a soft mask).
struct MaskedArray {
  ndarray data;
  ndarray mask;
  bool hard = false;  // soft by default
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

// Hard / soft mask control (numpy.ma). With a hard mask, assigning a value
// through a masked position is a no-op (the mask protects it). With a soft mask
// (the default), assigning an unmasked value writes it and clears the mask at
// that position. harden_mask/soften_mask toggle the mode; hardmask reports it.
NUMPP_API void harden_mask(MaskedArray& m);
NUMPP_API void soften_mask(MaskedArray& m);
NUMPP_API bool hardmask(const MaskedArray& m);

// Assign `value` at multi-index `idx`, honoring the hard/soft mask semantics
// above (numpy.ma item assignment).
NUMPP_API void setitem(MaskedArray& m, const std::vector<int64_t>& idx, double value);

}  // namespace ma
}  // namespace numpp
