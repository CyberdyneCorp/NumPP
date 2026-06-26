#pragma once

#include <string>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace npchar {

struct Partition3 {
  ndarray before;
  ndarray sep;
  ndarray after;
};

// Join each element's characters with `sep` (numpy.char.join(sep, a)).
NUMPP_API ndarray join(const std::string& sep, const ndarray& a);
// Encode 'U' -> 'S' (ASCII) and decode 'S' -> 'U'.
NUMPP_API ndarray encode(const ndarray& a);
NUMPP_API ndarray decode(const ndarray& a);
// Split each element at the first occurrence of `sep` (numpy.char.partition).
NUMPP_API Partition3 partition(const ndarray& a, const std::string& sep);

}  // namespace npchar
}  // namespace numpp
