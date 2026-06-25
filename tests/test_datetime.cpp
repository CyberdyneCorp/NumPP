#include <filesystem>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;
namespace fs = std::filesystem;

static std::string pyrepr(const std::string& code) {
  std::string out, cmd = "import numpy as np,sys; a=" + code + "; sys.stdout.write(repr(a))";
  if (FILE* f = popen(("python3 -c \"" + cmd + "\"").c_str(), "r")) { char b[4096]; size_t n; while ((n = fread(b, 1, sizeof(b), f)) > 0) out.append(b, n); pclose(f); }
  return out;
}

TEST_CASE("datetime64[D] parse and epoch value matches numpy") {
  ndarray a = datetime_array({"2020-01-01", "2020-01-03", "1970-01-01"}, 'D');
  CHECK(a.dtype().kind() == 'M');
  CHECK(dt_get(a, 0) == 18262);   // numpy datetime64('2020-01-01','D').astype(int)
  CHECK(dt_get(a, 2) == 0);
  CHECK(format_datetime(a.dtype(), dt_get(a, 0)) == "2020-01-01");
}

TEST_CASE("datetime64[s] parse with time") {
  ndarray a = datetime_array({"2020-01-01T12:00:00"}, 's');
  CHECK(dt_get(a, 0) == 1577880000);
  CHECK(format_datetime(a.dtype(), 1577880000) == "2020-01-01T12:00:00");
}

TEST_CASE("datetime arithmetic") {
  ndarray a = datetime_array({"2020-01-10", "2020-02-01"}, 'D');
  ndarray b = datetime_array({"2020-01-01", "2020-01-01"}, 'D');
  ndarray d = dt_subtract(a, b);     // timedelta in days
  CHECK(d.dtype().kind() == 'm');
  CHECK(dt_get(d, 0) == 9);
  CHECK(dt_get(d, 1) == 31);
  ndarray t = timedelta_array({5, 5}, 'D');
  ndarray sum = dt_add(b, t);        // datetime + timedelta
  CHECK(sum.dtype().kind() == 'M');
  CHECK(format_datetime(sum.dtype(), dt_get(sum, 0)) == "2020-01-06");
  CHECK(dt_less(b, a).item<bool>({0}) == true);
  CHECK(dt_equal(a, a).item<bool>({1}) == true);
}

TEST_CASE("datetime repr matches numpy") {
  ndarray a = datetime_array({"2020-01-01", "2020-01-03"}, 'D');
  if (npt::numpy_available())
    CHECK(array_repr(a) == pyrepr("np.array(['2020-01-01','2020-01-03'],dtype='datetime64[D]')"));
  CHECK(array_str(a) == "['2020-01-01' '2020-01-03']");
}

TEST_CASE("datetime npy round trip (numpp <-> numpy)") {
  ndarray a = datetime_array({"2021-06-15", "2000-02-29"}, 'D');
  std::string path = (fs::temp_directory_path() / "numpp_dt.npy").string();
  save(path, a);
  ndarray b = load(path);
  CHECK(b.dtype() == a.dtype());
  CHECK(dt_get(b, 0) == dt_get(a, 0));
  CHECK(dt_get(b, 1) == dt_get(a, 1));
  if (npt::numpy_available()) {
    std::string code = "import numpy as np; d=np.load(r'" + path + "'); "
                       "assert str(d[0])=='2021-06-15' and str(d[1])=='2000-02-29', d";
    CHECK(std::system(("python3 -c \"" + code + "\" 2>/dev/null").c_str()) == 0);
  }
  std::error_code ec; fs::remove(path, ec);
}
