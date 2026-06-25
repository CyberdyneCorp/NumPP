#pragma once

#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// datetime64/timedelta64 dtypes carry a unit char (D, s, ...) in DType metadata.
NUMPP_API DType make_datetime(char unit);
NUMPP_API DType make_timedelta(char unit);

NUMPP_API ndarray datetime_array(const std::vector<std::string>& iso, char unit);
NUMPP_API ndarray timedelta_array(const std::vector<int64_t>& vals, char unit);

NUMPP_API int64_t dt_get(const ndarray& a, int64_t i);
NUMPP_API void dt_set(ndarray& a, int64_t i, int64_t v);
NUMPP_API std::string format_datetime(DType dt, int64_t value);  // ISO for 'M', integer for 'm'

// datetime-datetime -> timedelta; datetime+timedelta -> datetime; comparisons.
NUMPP_API ndarray dt_subtract(const ndarray& a, const ndarray& b);
NUMPP_API ndarray dt_add(const ndarray& a, const ndarray& b);
NUMPP_API ndarray dt_equal(const ndarray& a, const ndarray& b);
NUMPP_API ndarray dt_less(const ndarray& a, const ndarray& b);

}  // namespace numpp
