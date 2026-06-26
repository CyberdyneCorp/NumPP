#include "numpp/datetime/busday.hpp"

#include "numpp/datetime/datetime.hpp"
#include "numpp/core/creation.hpp"
#include "numpp/core/error.hpp"
#include <string>
#include <vector>


namespace numpp {

namespace {

// Weekday of a day count (days since 1970-01-01, a Thursday).
// 0=Sun,1=Mon,...,6=Sat -> day 0 maps to Thu=4.
inline int weekday(int64_t day) { return static_cast<int>(((day % 7) + 7 + 4) % 7); }

inline bool is_business(int64_t day) {
  const int d = weekday(day);
  return d >= 1 && d <= 5;
}

// Business days in the half-open interval [a, b) for a <= b.
int64_t count_busdays_nonneg(int64_t a, int64_t b) {
  const int64_t total = b - a;
  const int64_t weeks = total / 7;
  const int64_t rem = total % 7;
  int64_t count = weeks * 5;
  const int wa = weekday(a);
  for (int64_t j = 0; j < rem; ++j) {
    const int d = (wa + static_cast<int>(j)) % 7;
    if (d >= 1 && d <= 5) ++count;
  }
  return count;
}

int64_t count_busdays(int64_t begin, int64_t end) {
  if (end >= begin) return count_busdays_nonneg(begin, end);
  return -count_busdays_nonneg(end, begin);
}

// Roll a day to a valid business day according to the roll convention.
int64_t roll_date(int64_t day, const std::string& roll) {
  if (is_business(day)) return day;
  if (roll == "forward" || roll == "following") {
    while (!is_business(day)) ++day;
  } else if (roll == "backward" || roll == "preceding") {
    while (!is_business(day)) --day;
  } else {
    // "raise" (or unknown): the date must already be a business day.
    throw value_error("Non-business day date in busday_offset with roll='raise'");
  }
  return day;
}

}  // namespace

ndarray is_busday(const ndarray& dates) {
  ndarray out(dates.shape(), kBool, Order::C);
  const int64_t n = dates.size();
  bool* o = n ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = is_business(dt_get(dates, i));
  return out;
}

ndarray busday_count(const ndarray& begindates, const ndarray& enddates) {
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
    o[i] = count_busdays(b, e);
  }
  return out;
}

ndarray busday_offset(const ndarray& dates, int64_t offsets, const std::string& roll) {
  ndarray out(dates.shape(), dates.dtype(), Order::C);
  const int64_t n = dates.size();
  for (int64_t i = 0; i < n; ++i) {
    int64_t day = roll_date(dt_get(dates, i), roll);
    int64_t rem = offsets;
    while (rem > 0) {
      ++day;
      if (is_business(day)) --rem;
    }
    while (rem < 0) {
      --day;
      if (is_business(day)) ++rem;
    }
    dt_set(out, i, day);
  }
  return out;
}

std::vector<std::string> datetime_as_string(const ndarray& a) {
  std::vector<std::string> out;
  const int64_t n = a.size();
  out.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out.push_back(format_datetime(a.dtype(), dt_get(a, i)));
  return out;
}

}  // namespace numpp
