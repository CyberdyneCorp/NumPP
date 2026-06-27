#pragma once

#include <array>
#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// A custom business-day calendar (numpy.busdaycalendar): a weekmask selecting
// which weekdays are valid and a holiday list to exclude. The weekmask is seven
// flags Monday..Sunday; the default "1111100" is Mon-Fri.
struct NUMPP_API busdaycalendar {
  std::array<bool, 7> weekmask{{true, true, true, true, true, false, false}};  // Mon..Sun
  std::vector<int64_t> holidays;  // day counts since 1970-01-01, sorted/unique,
                                  // with non-weekday entries removed (numpy rule)

  busdaycalendar() = default;
  // weekmask as a 7-char "1111100" string; holidays as day counts since the epoch.
  explicit busdaycalendar(const std::string& weekmask_str,
                          std::vector<int64_t> holiday_days = {});
  explicit busdaycalendar(const std::array<bool, 7>& mask,
                          std::vector<int64_t> holiday_days = {});
};

// Business-day calendar functions over datetime64[D] arrays. Dates are day
// counts since 1970-01-01 (a Thursday). The no-calendar overloads use the
// default Mon-Fri week with no holidays, matching numpy.busday_*.
NUMPP_API ndarray is_busday(const ndarray& dates);                         // bool array
NUMPP_API ndarray is_busday(const ndarray& dates, const busdaycalendar& cal);

NUMPP_API ndarray busday_count(const ndarray& begindates, const ndarray& enddates);  // int64, [begin, end)
NUMPP_API ndarray busday_count(const ndarray& begindates, const ndarray& enddates,
                               const busdaycalendar& cal);

// roll: "forward"/"following", "backward"/"preceding", "raise". Offset in business days.
NUMPP_API ndarray busday_offset(const ndarray& dates, int64_t offsets, const std::string& roll = "raise");
NUMPP_API ndarray busday_offset(const ndarray& dates, int64_t offsets, const std::string& roll,
                                const busdaycalendar& cal);

// Vectorized ISO-8601 formatting (numpy.datetime_as_string).
NUMPP_API std::vector<std::string> datetime_as_string(const ndarray& a);

}  // namespace numpp
