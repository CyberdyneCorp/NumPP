#pragma once

#include <string>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// numpy-compatible text formatting.
NUMPP_API std::string array_str(const ndarray& a);   // str(a): space-separated
NUMPP_API std::string array_repr(const ndarray& a);  // repr(a): 'array([...])' with dtype suffix
inline std::string to_string(const ndarray& a) { return array_str(a); }

}  // namespace numpp
