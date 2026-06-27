#include "numpp/datetime/busday.hpp"

#include "numpp/datetime/datetime.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <algorithm>
#include <string>
#include <vector>


namespace numpp {

namespace {

// Weekday of a day count (days since 1970-01-01, a Thursday).
// 0=Sun,1=Mon,...,6=Sat -> day 0 maps to Thu=4.
inline int weekday(int64_t day) { return static_cast<int>(((day % 7) + 7 + 4) % 7); }

// numpy weekmask index Mon..Sun (Mon=0) from a day count.
inline int weekmask_index(int64_t day) { return (weekday(day) + 6) % 7; }

inline bool is_holiday(int64_t day, const busdaycalendar& cal) {
  return std::binary_search(cal.holidays.begin(), cal.holidays.end(), day);
}

inline bool is_business(int64_t day, const busdaycalendar& cal) {
  return cal.weekmask[weekmask_index(day)] && !is_holiday(day, cal);
}

// Business days in the half-open interval [begin, end), signed. Iterates so any
// weekmask/holiday combination is handled exactly.
int64_t count_busdays(int64_t begin, int64_t end, const busdaycalendar& cal) {
  if (end == begin) return 0;
  const int64_t lo = begin < end ? begin : end;
  const int64_t hi = begin < end ? end : begin;
  int64_t count = 0;
  for (int64_t day = lo; day < hi; ++day)
    if (is_business(day, cal)) ++count;
  return end >= begin ? count : -count;
}

// Roll a day to a valid business day according to the roll convention.
int64_t roll_date(int64_t day, const std::string& roll, const busdaycalendar& cal) {
  if (is_business(day, cal)) return day;
  if (roll == "forward" || roll == "following") {
    while (!is_business(day, cal)) ++day;
  } else if (roll == "backward" || roll == "preceding") {
    while (!is_business(day, cal)) --day;
  } else {
    // "raise" (or unknown): the date must already be a business day.
    throw value_error("Non-business day date in busday_offset with roll='raise'");
  }
  return day;
}

const busdaycalendar& default_calendar() {
  static const busdaycalendar cal;
  return cal;
}

}  // namespace

busdaycalendar::busdaycalendar(const std::string& weekmask_str, std::vector<int64_t> holiday_days) {
  if (weekmask_str.size() != 7)
    throw value_error("busdaycalendar weekmask must be a 7-character '1111100' string");
  for (int i = 0; i < 7; ++i) weekmask[static_cast<size_t>(i)] = weekmask_str[static_cast<size_t>(i)] == '1';
  // numpy drops holidays that fall on a non-business weekday, then sorts/uniques.
  for (int64_t d : holiday_days)
    if (weekmask[static_cast<size_t>(weekmask_index(d))]) holidays.push_back(d);
  std::sort(holidays.begin(), holidays.end());
  holidays.erase(std::unique(holidays.begin(), holidays.end()), holidays.end());
}

busdaycalendar::busdaycalendar(const std::array<bool, 7>& mask, std::vector<int64_t> holiday_days)
    : weekmask(mask) {
  for (int64_t d : holiday_days)
    if (weekmask[static_cast<size_t>(weekmask_index(d))]) holidays.push_back(d);
  std::sort(holidays.begin(), holidays.end());
  holidays.erase(std::unique(holidays.begin(), holidays.end()), holidays.end());
}

ndarray is_busday(const ndarray& dates, const busdaycalendar& cal) {
  ndarray out(dates.shape(), kBool, Order::C);
  const int64_t n = dates.size();
  bool* o = n ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = is_business(dt_get(dates, i), cal);
  return out;
}
ndarray is_busday(const ndarray& dates) { return is_busday(dates, default_calendar()); }

ndarray busday_count(const ndarray& begindates, const ndarray& enddates, const busdaycalendar& cal) {
  const int64_t nb = begindates.size();
  const int64_t ne = enddates.size();
  if (nb != ne && nb != 1 && ne != 1)
    throw value_error("busday_count: operands could not be broadcast together");
  const int64_t n = nb > ne ? nb : ne;
  ndarray out(Shape{n}, kInt64, Order::C);
  int64_t* o = n ? out.typed_data<int64_t>() : nullptr;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t b = dt_get(begindates, nb == 1 ? 0 : i);
    const int64_t e = dt_get(enddates, ne == 1 ? 0 : i);
    o[i] = count_busdays(b, e, cal);
  }
  return out;
}
ndarray busday_count(const ndarray& begindates, const ndarray& enddates) {
  return busday_count(begindates, enddates, default_calendar());
}

ndarray busday_offset(const ndarray& dates, int64_t offsets, const std::string& roll,
                      const busdaycalendar& cal) {
  ndarray out(dates.shape(), dates.dtype(), Order::C);
  const int64_t n = dates.size();
  for (int64_t i = 0; i < n; ++i) {
    int64_t day = roll_date(dt_get(dates, i), roll, cal);
    int64_t rem = offsets;
    while (rem > 0) {
      ++day;
      if (is_business(day, cal)) --rem;
    }
    while (rem < 0) {
      --day;
      if (is_business(day, cal)) ++rem;
    }
    dt_set(out, i, day);
  }
  return out;
}
ndarray busday_offset(const ndarray& dates, int64_t offsets, const std::string& roll) {
  return busday_offset(dates, offsets, roll, default_calendar());
}

std::vector<std::string> datetime_as_string(const ndarray& a) {
  std::vector<std::string> out;
  const int64_t n = a.size();
  out.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out.push_back(format_datetime(a.dtype(), dt_get(a, i)));
  return out;
}

}  // namespace numpp
