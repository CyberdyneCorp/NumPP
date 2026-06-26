#pragma once

#include <cstdint>
#include <string>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace npchar {

// Vectorized string operations over 1-D 'U'/'S' string arrays, mirroring
// numpy.char.*. String-returning ops return a new string array; predicate /
// position ops return bool / int64 arrays.

// ---- string -> string ----
NUMPP_API ndarray add(const ndarray& a, const ndarray& b);
NUMPP_API ndarray multiply(const ndarray& a, int64_t n);
NUMPP_API ndarray upper(const ndarray& a);
NUMPP_API ndarray lower(const ndarray& a);
NUMPP_API ndarray capitalize(const ndarray& a);
NUMPP_API ndarray title(const ndarray& a);
NUMPP_API ndarray strip(const ndarray& a);
NUMPP_API ndarray lstrip(const ndarray& a);
NUMPP_API ndarray rstrip(const ndarray& a);
NUMPP_API ndarray replace(const ndarray& a, const std::string& old_s, const std::string& new_s);

// ---- string -> int64 / bool ----
NUMPP_API ndarray str_len(const ndarray& a);
NUMPP_API ndarray find(const ndarray& a, const std::string& sub);
NUMPP_API ndarray count(const ndarray& a, const std::string& sub);
NUMPP_API ndarray startswith(const ndarray& a, const std::string& prefix);
NUMPP_API ndarray endswith(const ndarray& a, const std::string& suffix);

}  // namespace npchar
}  // namespace numpp
