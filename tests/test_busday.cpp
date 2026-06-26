#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace numpp;

namespace {
// Build a datetime64[D] array from consecutive day counts, matching
// np.arange('start','stop',dtype='datetime64[D]').
ndarray dt_range(const std::string& start, const std::string& stop) {
  const int64_t a = dt_get(datetime_array({start}, 'D'), 0);
  const int64_t b = dt_get(datetime_array({stop}, 'D'), 0);
  ndarray out(Shape{b - a}, make_datetime('D'), Order::C);
  for (int64_t i = 0; i < b - a; ++i) dt_set(out, i, a + i);
  return out;
}
}  // namespace

TEST_CASE("is_busday vs numpy") {
  ndarray d = dt_range("2011-07-01", "2011-07-12");  // Fri..next Mon, spans a weekend
  auto o = npt::oracle("a=np.is_busday(np.arange('2011-07-01','2011-07-12',dtype='datetime64[D]'))");
  if (o) CHECK(allclose(is_busday(d), *o, 1e-9, 1e-12, true));
}

TEST_CASE("is_busday explicit weekend") {
  ndarray d = dt_range("2011-07-01", "2011-07-12");
  ndarray b = is_busday(d);
  // 2011-07-01 Fri true, 07-02 Sat false, 07-03 Sun false, 07-04 Mon true.
  CHECK((b.item<bool>({0}) == true));
  CHECK((b.item<bool>({1}) == false));
  CHECK((b.item<bool>({2}) == false));
  CHECK((b.item<bool>({3}) == true));
}

TEST_CASE("busday_count single pair vs numpy") {
  ndarray b = datetime_array({"2011-07-01"}, 'D');
  ndarray e = datetime_array({"2011-07-12"}, 'D');
  auto o = npt::oracle("a=np.array([np.busday_count(np.datetime64('2011-07-01'),np.datetime64('2011-07-12'))])");
  if (o) CHECK(allclose(busday_count(b, e), *o, 1e-9, 1e-12, true));
}

TEST_CASE("busday_count full year vs numpy") {
  ndarray b = datetime_array({"2011-01-01"}, 'D');
  ndarray e = datetime_array({"2012-01-01"}, 'D');
  auto o = npt::oracle("a=np.array([np.busday_count(np.datetime64('2011-01-01'),np.datetime64('2012-01-01'))])");
  if (o) CHECK(allclose(busday_count(b, e), *o, 1e-9, 1e-12, true));
}

TEST_CASE("busday_count negative (end before begin) vs numpy") {
  ndarray b = datetime_array({"2011-07-12"}, 'D');
  ndarray e = datetime_array({"2011-07-01"}, 'D');
  auto o = npt::oracle("a=np.array([np.busday_count(np.datetime64('2011-07-12'),np.datetime64('2011-07-01'))])");
  if (o) CHECK(allclose(busday_count(b, e), *o, 1e-9, 1e-12, true));
}

TEST_CASE("busday_count array vs numpy") {
  ndarray b = datetime_array({"2011-07-01", "2011-08-01", "2011-09-01"}, 'D');
  ndarray e = datetime_array({"2011-07-31", "2011-08-31", "2011-09-30"}, 'D');
  auto o = npt::oracle(
      "a=np.busday_count(np.array(['2011-07-01','2011-08-01','2011-09-01'],dtype='datetime64[D]'),"
      "np.array(['2011-07-31','2011-08-31','2011-09-30'],dtype='datetime64[D]'))");
  if (o) CHECK(allclose(busday_count(b, e), *o, 1e-9, 1e-12, true));
}

TEST_CASE("busday_offset forward roll vs numpy") {
  // 2011-07-02 is a Saturday; roll forward then +3 business days.
  ndarray d = datetime_array({"2011-07-02"}, 'D');
  ndarray r = busday_offset(d, 3, "forward");
  auto o = npt::oracle("a=np.array([np.busday_offset('2011-07-02',3,roll='forward')])");
  if (o) CHECK(dt_get(r, 0) == dt_get(*o, 0));
  CHECK(datetime_as_string(r)[0] == "2011-07-07");  // Sat->Mon 07-04, +3 -> Thu 07-07
}

TEST_CASE("busday_offset backward roll vs numpy") {
  ndarray d = datetime_array({"2011-07-03"}, 'D');  // Sunday
  ndarray r = busday_offset(d, 1, "backward");
  auto o = npt::oracle("a=np.array([np.busday_offset('2011-07-03',1,roll='backward')])");
  if (o) CHECK(dt_get(r, 0) == dt_get(*o, 0));
}

TEST_CASE("busday_offset zero offset on business day vs numpy") {
  ndarray d = datetime_array({"2011-07-01"}, 'D');  // Friday, already business
  ndarray r = busday_offset(d, 0, "raise");
  auto o = npt::oracle("a=np.array([np.busday_offset('2011-07-01',0,roll='raise')])");
  if (o) CHECK(dt_get(r, 0) == dt_get(*o, 0));
}

TEST_CASE("busday_offset negative offset vs numpy") {
  ndarray d = datetime_array({"2011-07-15"}, 'D');  // Friday
  ndarray r = busday_offset(d, -5, "forward");
  auto o = npt::oracle("a=np.array([np.busday_offset('2011-07-15',-5,roll='forward')])");
  if (o) CHECK(dt_get(r, 0) == dt_get(*o, 0));
}

TEST_CASE("busday_offset raise on weekend throws") {
  ndarray d = datetime_array({"2011-07-02"}, 'D');  // Saturday
  CHECK_THROWS_AS(busday_offset(d, 1, "raise"), numpp::value_error);
}

TEST_CASE("datetime_as_string exact values") {
  ndarray a = datetime_array({"2011-07-01", "2020-01-01", "1970-01-01", "2000-02-29"}, 'D');
  std::vector<std::string> got = datetime_as_string(a);
  std::vector<std::string> want = {"2011-07-01", "2020-01-01", "1970-01-01", "2000-02-29"};
  CHECK(got == want);
}

TEST_CASE("datetime_as_string seconds unit") {
  ndarray a = datetime_array({"2020-01-01T12:00:00"}, 's');
  CHECK(datetime_as_string(a)[0] == "2020-01-01T12:00:00");
}
