#include "numpp/datetime/datetime.hpp"

#include <cstdio>
#include <cstring>

namespace numpp {
namespace {

// Howard Hinnant's civil<->days (days since 1970-01-01).
int64_t days_from_civil(int64_t y, int64_t m, int64_t d) {
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const int64_t yoe = y - era * 400;
  const int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + doe - 719468;
}
void civil_from_days(int64_t z, int64_t& y, int64_t& m, int64_t& d) {
  z += 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const int64_t doe = z - era * 146097;
  const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  y = yoe + era * 400;
  const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const int64_t mp = (5 * doy + 2) / 153;
  d = doy - (153 * mp + 2) / 5 + 1;
  m = mp + (mp < 10 ? 3 : -9);
  y += m <= 2;
}

int64_t parse_iso(const std::string& s, char unit) {
  int64_t y = std::stoll(s.substr(0, 4)), mo = std::stoll(s.substr(5, 2)), d = std::stoll(s.substr(8, 2));
  int64_t days = days_from_civil(y, mo, d);
  if (unit == 'D') return days;
  int64_t hh = 0, mm = 0, ss = 0;
  if (s.size() >= 19 && s[10] == 'T') { hh = std::stoll(s.substr(11, 2)); mm = std::stoll(s.substr(14, 2)); ss = std::stoll(s.substr(17, 2)); }
  return days * 86400 + hh * 3600 + mm * 60 + ss;  // unit 's'
}

char unit_of(DType d) { return d.meta() ? d.meta()->unit : 'D'; }
int64_t* int_data(ndarray& a) { return reinterpret_cast<int64_t*>(a.bytes()); }
const int64_t* int_data(const ndarray& a) { return reinterpret_cast<const int64_t*>(a.bytes()); }

}  // namespace

DType make_datetime(char unit) { auto m = std::make_shared<DTypeMeta>(); m->itemsize = 8; m->unit = unit; return DType(DTypeId::Datetime64, m); }
DType make_timedelta(char unit) { auto m = std::make_shared<DTypeMeta>(); m->itemsize = 8; m->unit = unit; return DType(DTypeId::Timedelta64, m); }

ndarray datetime_array(const std::vector<std::string>& iso, char unit) {
  ndarray a(Shape{static_cast<int64_t>(iso.size())}, make_datetime(unit), Order::C);
  int64_t* p = a.size() ? int_data(a) : nullptr;
  for (size_t i = 0; i < iso.size(); ++i) p[i] = parse_iso(iso[i], unit);
  return a;
}
ndarray timedelta_array(const std::vector<int64_t>& vals, char unit) {
  ndarray a(Shape{static_cast<int64_t>(vals.size())}, make_timedelta(unit), Order::C);
  int64_t* p = a.size() ? int_data(a) : nullptr;
  for (size_t i = 0; i < vals.size(); ++i) p[i] = vals[i];
  return a;
}

int64_t dt_get(const ndarray& a, int64_t i) { return int_data(a)[i]; }
void dt_set(ndarray& a, int64_t i, int64_t v) { int_data(a)[i] = v; }

std::string format_datetime(DType dt, int64_t value) {
  if (dt.kind() == 'm') return std::to_string(value);  // timedelta prints as integer
  const char unit = unit_of(dt);
  int64_t days = unit == 'D' ? value : value / 86400;
  int64_t rem = unit == 'D' ? 0 : value - days * 86400;
  if (rem < 0) { rem += 86400; --days; }
  int64_t y, mo, d;
  civil_from_days(days, y, mo, d);
  // Sized for the worst case: a full-range 64-bit year (up to 20 chars) plus the
  // "-MM-DDTHH:MM:SS" tail, so snprintf can never truncate (silences
  // -Wformat-truncation under GCC -O FORTIFY; see #109).
  char buf[96];
  if (unit == 'D') std::snprintf(buf, sizeof(buf), "%04lld-%02lld-%02lld", (long long)y, (long long)mo, (long long)d);
  else std::snprintf(buf, sizeof(buf), "%04lld-%02lld-%02lldT%02lld:%02lld:%02lld", (long long)y, (long long)mo, (long long)d,
                     (long long)(rem / 3600), (long long)((rem % 3600) / 60), (long long)(rem % 60));
  return buf;
}

ndarray dt_subtract(const ndarray& a, const ndarray& b) {
  ndarray out(a.shape(), make_timedelta(unit_of(a.dtype())), Order::C);
  const int64_t n = a.size();
  const int64_t *pa = int_data(a), *pb = int_data(b);
  int64_t* o = n ? int_data(out) : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = pa[i] - pb[i];
  return out;
}
ndarray dt_add(const ndarray& a, const ndarray& b) {  // datetime + timedelta
  const bool a_dt = a.dtype().kind() == 'M';
  ndarray out(a.shape(), make_datetime(unit_of(a_dt ? a.dtype() : b.dtype())), Order::C);
  const int64_t n = a.size();
  const int64_t *pa = int_data(a), *pb = int_data(b);
  int64_t* o = n ? int_data(out) : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = pa[i] + pb[i];
  return out;
}
ndarray dt_equal(const ndarray& a, const ndarray& b) {
  ndarray out(a.shape(), kBool, Order::C);
  const int64_t n = a.size();
  const int64_t *pa = int_data(a), *pb = int_data(b);
  bool* o = n ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = pa[i] == pb[i];
  return out;
}
ndarray dt_less(const ndarray& a, const ndarray& b) {
  ndarray out(a.shape(), kBool, Order::C);
  const int64_t n = a.size();
  const int64_t *pa = int_data(a), *pb = int_data(b);
  bool* o = n ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = pa[i] < pb[i];
  return out;
}

}  // namespace numpp
