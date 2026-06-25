#include <filesystem>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;
namespace fs = std::filesystem;

TEST_CASE("string array creation, dtype, get/set") {
  ndarray a = string_array({"ab", "c", "hello"});
  CHECK(a.dtype().kind() == 'U');
  CHECK(a.dtype().itemsize() == 20);   // 5 chars * 4 bytes
  CHECK(a.shape() == Shape({3}));
  CHECK(get_string(a, 0) == "ab");
  CHECK(get_string(a, 1) == "c");
  CHECK(get_string(a, 2) == "hello");
  set_string(a, 1, "xyz");
  CHECK(get_string(a, 1) == "xyz");
}

TEST_CASE("fixed-width truncation and bytes") {
  ndarray u = string_array({"abcdef", "g"}, 3);
  CHECK(get_string(u, 0) == "abc");    // truncated to 3
  ndarray b = bytes_array({"hi", "there"});
  CHECK(b.dtype().kind() == 'S');
  CHECK(b.dtype().itemsize() == 5);
  CHECK(get_string(b, 1) == "there");
}

TEST_CASE("string equality") {
  ndarray a = string_array({"ab", "c", "x"}, 2);
  ndarray b = string_array({"ab", "d", "x"}, 2);
  ndarray e = str_equal(a, b);
  CHECK(e.dtype() == kBool);
  CHECK(e.item<bool>({0}) == true);
  CHECK(e.item<bool>({1}) == false);
  CHECK(e.item<bool>({2}) == true);
  CHECK(str_not_equal(a, b).item<bool>({1}) == true);
}

TEST_CASE("string str/repr vs numpy") {
  ndarray a = string_array({"ab", "c"}, 2);
  if (npt::numpy_available()) {
    auto pyrepr = [](const char* code) {
      std::string out; std::string cmd = "import numpy as np,sys; a=" + std::string(code) + "; sys.stdout.write(repr(a))";
      if (FILE* f = popen(("python3 -c \"" + cmd + "\"").c_str(), "r")) { char b[4096]; size_t n; while ((n = fread(b, 1, sizeof(b), f)) > 0) out.append(b, n); pclose(f); }
      return out;
    };
    CHECK(array_repr(a) == pyrepr("np.array(['ab','c'],dtype='U2')"));
  } else std::fprintf(stderr, "  [skip] string repr\n");
  CHECK(array_str(a) == "['ab' 'c']");
}

TEST_CASE("npy round trip for string arrays (numpp<->numpy)") {
  ndarray a = string_array({"alpha", "beta", "g"}, 5);
  std::string path = (fs::temp_directory_path() / "numpp_str.npy").string();
  save(path, a);
  ndarray b = load(path);
  CHECK(b.dtype() == a.dtype());
  for (int64_t i = 0; i < 3; ++i) CHECK(get_string(b, i) == get_string(a, i));
  if (npt::numpy_available()) {
    auto o = npt::oracle("a=np.load(r'" + path + "').astype('U5')");
    // compare values via a re-save by numpy is complex; just confirm numpy can read it
    std::string code = "import numpy as np; d=np.load(r'" + path + "'); assert list(d)==['alpha','beta','g'], list(d)";
    int rc = std::system(("python3 -c \"" + code + "\" 2>/dev/null").c_str());
    CHECK(rc == 0);
  }
  std::error_code ec; fs::remove(path, ec);
}
