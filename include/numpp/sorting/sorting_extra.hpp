#pragma once

#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Indirect stable sort using a sequence of keys; the LAST key is primary (numpy).
NUMPP_API ndarray lexsort(const std::vector<ndarray>& keys);

// Sort a complex array by real part, then imaginary part (returns complex128).
NUMPP_API ndarray sort_complex(const ndarray& a);

// searchsorted into an unsorted `a` whose sort order is given by `sorter`.
NUMPP_API ndarray searchsorted(const ndarray& a, const ndarray& v, const std::string& side,
                               const ndarray& sorter);

}  // namespace numpp
