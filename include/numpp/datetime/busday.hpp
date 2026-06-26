#pragma once

#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Business-day calendar functions over datetime64[D] arrays (Mon-Fri week),
// matching numpy.busday_*. Dates are day counts since 1970-01-01 (a Thursday).
NUMPP_API ndarray is_busday(const ndarray& dates);                         // bool array
NUMPP_API ndarray busday_count(const ndarray& begindates, const ndarray& enddates);  // int64, [begin, end)
// roll: "forward"/"following", "backward"/"preceding". Offset in business days.
NUMPP_API ndarray busday_offset(const ndarray& dates, int64_t offsets, const std::string& roll = "raise");

// Vectorized ISO-8601 formatting (numpy.datetime_as_string).
NUMPP_API std::vector<std::string> datetime_as_string(const ndarray& a);

}  // namespace numpp
