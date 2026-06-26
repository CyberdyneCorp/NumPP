#pragma once

#include <cstdint>
#include <string>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace npchar {

// More numpy.char vectorized ops over 1-D string arrays.

// ---- string -> string ----
NUMPP_API ndarray center(const ndarray& a, int64_t width, char fillchar = ' ');
NUMPP_API ndarray ljust(const ndarray& a, int64_t width, char fillchar = ' ');
NUMPP_API ndarray rjust(const ndarray& a, int64_t width, char fillchar = ' ');
NUMPP_API ndarray zfill(const ndarray& a, int64_t width);
NUMPP_API ndarray swapcase(const ndarray& a);
NUMPP_API ndarray expandtabs(const ndarray& a, int64_t tabsize = 8);

// ---- string -> bool ----
NUMPP_API ndarray isalpha(const ndarray& a);
NUMPP_API ndarray isdigit(const ndarray& a);
NUMPP_API ndarray isspace(const ndarray& a);
NUMPP_API ndarray isupper(const ndarray& a);
NUMPP_API ndarray islower(const ndarray& a);
NUMPP_API ndarray isalnum(const ndarray& a);
NUMPP_API ndarray istitle(const ndarray& a);

}  // namespace npchar
}  // namespace numpp
