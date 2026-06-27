#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <string>
#include <vector>

using namespace numpp;

namespace {
ndarray dt_range(const std::string& start, const std::string& stop) {
  const int64_t a = dt_get(datetime_array({start}, 'D'), 0);
  const int64_t b = dt_get(datetime_array({stop}, 'D'), 0);
  ndarray out(Shape{b - a}, make_datetime('D'), Order::C);
  for (int64_t i = 0; i < b - a; ++i) dt_set(out, i, a + i);
  return out;
}
int64_t day_of(const std::string& s) { return dt_get(datetime_array({s}, 'D'), 0); }
}  // namespace

TEST_CASE("is_busday with custom weekmask (Sat working) vs numpy") {
  ndarray d = dt_range("2011-07-01", "2011-07-12");
  busdaycalendar cal("1111110");  // Mon..Sat valid, Sun off
  auto o = npt::oracle(
      "a=np.is_busday(np.arange('2011-07-01','2011-07-12',dtype='datetime64[D]'),"
      "weekmask='1111110')");
  if (o) CHECK(allclose(is_busday(d, cal), *o, 1e-9, 1e-12, true));
}

TEST_CASE("is_busday with a holiday vs numpy") {
  ndarray d = dt_range("2011-07-01", "2011-07-12");
  busdaycalendar cal("1111100", {day_of("2011-07-04")});  // Mon 07-04 is a holiday
  auto o = npt::oracle(
      "a=np.is_busday(np.arange('2011-07-01','2011-07-12',dtype='datetime64[D]'),"
      "holidays=np.array(['2011-07-04'],dtype='datetime64[D]'))");
  if (o) CHECK(allclose(is_busday(d, cal), *o, 1e-9, 1e-12, true));
}

TEST_CASE("is_busday explicit: holiday makes Monday non-business") {
  ndarray d = dt_range("2011-07-01", "2011-07-12");
  busdaycalendar cal("1111100", {day_of("2011-07-04")});
  ndarray b = is_busday(d, cal);
  CHECK((b.item<bool>({0}) == true));   // Fri 07-01
  CHECK((b.item<bool>({3}) == false));  // Mon 07-04 holiday
  CHECK((b.item<bool>({4}) == true));   // Tue 07-05
}

TEST_CASE("busday_count with weekmask+holidays vs numpy") {
  ndarray b = datetime_array({"2011-07-01"}, 'D');
  ndarray e = datetime_array({"2011-07-12"}, 'D');
  busdaycalendar cal("1111100", {day_of("2011-07-04")});
  auto o = npt::oracle(
      "a=np.array([np.busday_count(np.datetime64('2011-07-01'),np.datetime64('2011-07-12'),"
      "holidays=np.array(['2011-07-04'],dtype='datetime64[D]'))])");
  if (o) CHECK(allclose(busday_count(b, e, cal), *o, 1e-9, 1e-12, true));
}

TEST_CASE("busday_offset over a holiday with roll=forward vs numpy") {
  ndarray d = datetime_array({"2011-07-01"}, 'D');  // Friday
  busdaycalendar cal("1111100", {day_of("2011-07-04")});
  auto o = npt::oracle(
      "a=np.array([np.busday_offset(np.datetime64('2011-07-01'),1,roll='forward',"
      "holidays=np.array(['2011-07-04'],dtype='datetime64[D]'))]).astype('datetime64[D]').astype('int64')");
  // NumPP offset result is datetime64[D]; compare the raw day counts.
  ndarray got = busday_offset(d, 1, "forward", cal);
  ndarray got_i64(Shape{1}, kInt64, Order::C);
  got_i64.set_item<int64_t>({0}, dt_get(got, 0));
  if (o) CHECK(allclose(got_i64, *o, 1e-9, 1e-12, true));
}

TEST_CASE("busdaycalendar rejects a malformed weekmask") {
  CHECK_THROWS_AS(busdaycalendar("11111"), value_error);
}
