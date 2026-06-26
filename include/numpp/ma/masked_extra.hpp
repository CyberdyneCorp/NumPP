#pragma once

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"
#include "numpp/ma/masked.hpp"

namespace numpp {
namespace ma {

// ---- elementwise arithmetic (result mask = a.mask | b.mask, broadcast) ----
NUMPP_API MaskedArray add(const MaskedArray& a, const MaskedArray& b);
NUMPP_API MaskedArray subtract(const MaskedArray& a, const MaskedArray& b);
NUMPP_API MaskedArray multiply(const MaskedArray& a, const MaskedArray& b);
NUMPP_API MaskedArray divide(const MaskedArray& a, const MaskedArray& b);

// ---- comparison/range mask constructors ----
NUMPP_API MaskedArray masked_less(const ndarray& a, double value);
NUMPP_API MaskedArray masked_less_equal(const ndarray& a, double value);
NUMPP_API MaskedArray masked_greater_equal(const ndarray& a, double value);
NUMPP_API MaskedArray masked_not_equal(const ndarray& a, double value);
NUMPP_API MaskedArray masked_inside(const ndarray& a, double v1, double v2);
NUMPP_API MaskedArray masked_outside(const ndarray& a, double v1, double v2);
NUMPP_API MaskedArray masked_values(const ndarray& a, double value, double rtol = 1e-5, double atol = 1e-8);

// ---- per-axis reductions (return MaskedArray reduced along axis; an output
// element is masked iff every element of its slice was masked) ----
NUMPP_API MaskedArray sum_axis(const MaskedArray& m, int64_t axis);
NUMPP_API MaskedArray prod_axis(const MaskedArray& m, int64_t axis);
NUMPP_API MaskedArray mean_axis(const MaskedArray& m, int64_t axis);
NUMPP_API MaskedArray max_axis(const MaskedArray& m, int64_t axis);
NUMPP_API MaskedArray min_axis(const MaskedArray& m, int64_t axis);
NUMPP_API ndarray count_axis(const MaskedArray& m, int64_t axis);  // int64 unmasked counts

// ---- mask accessors ----
NUMPP_API ndarray getmask(const MaskedArray& m);   // bool mask
NUMPP_API ndarray getdata(const MaskedArray& m);   // underlying data

}  // namespace ma
}  // namespace numpp
