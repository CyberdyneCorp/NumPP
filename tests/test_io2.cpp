#include <filesystem>
#include <fstream>
#include <map>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;
namespace fs = std::filesystem;
static std::string tmp(const std::string& n) { return (fs::temp_directory_path() / n).string(); }

TEST_CASE("npz round trip in numpp") {
  std::vector<NamedArray> arrays = {
      {"a", arange(0.0, 6.0, 1.0).reshape({2, 3})},
      {"b", arange(1.0, 5.0, 1.0).astype(kInt32)},
      {"c", full({3}, 2.5, kFloat32)},
  };
  std::string path = tmp("numpp_test.npz");
  savez(path, arrays);
  auto loaded = load_npz(path);
  CHECK(loaded.size() == 3);
  CHECK(array_equal(loaded.at("a"), arrays[0].second));
  CHECK(array_equal(loaded.at("b"), arrays[1].second));
  CHECK(array_equal(loaded.at("c"), arrays[2].second));
  std::error_code ec; fs::remove(path, ec);
}

TEST_CASE("numpp savez -> numpy load matches") {
  if (!npt::numpy_available()) { std::fprintf(stderr, "  [skip]\n"); return; }
  std::vector<NamedArray> arrays = {{"x", arange(0.0, 12.0, 1.0).reshape({3, 4})},
                                    {"y", arange(0.0, 5.0, 1.0).astype(kInt64)}};
  std::string path = tmp("numpp_savez.npz");
  savez(path, arrays);
  auto ox = npt::oracle("d=np.load(r'" + path + "');a=d['x']");
  auto oy = npt::oracle("d=np.load(r'" + path + "');a=d['y']");
  std::error_code ec; fs::remove(path, ec);
  if (ox) CHECK(array_equal(arrays[0].second, *ox));
  if (oy) CHECK(array_equal(arrays[1].second, *oy));
}

TEST_CASE("numpy savez -> numpp load_npz matches") {
  if (!npt::numpy_available()) { std::fprintf(stderr, "  [skip]\n"); return; }
  std::string path = tmp("np_savez.npz"), pyf = tmp("np_savez.py");
  { std::ofstream s(pyf); s << "import numpy as np\n"
       << "np.savez(r'" << path << "', p=np.arange(8.0).reshape(2,4), q=np.array([1,2,3],dtype=np.int32))\n"; }
  if (std::system(("python3 " + pyf + " >/dev/null 2>&1").c_str()) != 0) { std::fprintf(stderr, "  [skip]\n"); return; }
  auto loaded = load_npz(path);
  CHECK(loaded.count("p") == 1 && loaded.count("q") == 1);
  CHECK(array_equal(loaded.at("p"), arange(0.0, 8.0, 1.0).reshape({2, 4})));
  CHECK(array_equal(loaded.at("q"), arange(1.0, 4.0, 1.0).astype(kInt32)));
  std::error_code ec; fs::remove(path, ec); fs::remove(pyf, ec);
}

TEST_CASE("savez_compressed readable by numpy") {
  if (!npt::numpy_available()) { std::fprintf(stderr, "  [skip]\n"); return; }
  std::string path = tmp("numpp_savezc.npz");
  savez_compressed(path, {{"z", arange(0.0, 100.0, 1.0)}});
  auto o = npt::oracle("d=np.load(r'" + path + "');a=d['z']");
  std::error_code ec; fs::remove(path, ec);
  if (o) CHECK(array_equal(arange(0.0, 100.0, 1.0), *o));
}
