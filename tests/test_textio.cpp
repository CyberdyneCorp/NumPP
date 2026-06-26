#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <filesystem>
#include <fstream>
#include <string>

using namespace numpp;
namespace fs = std::filesystem;

namespace {
std::string tmp(const std::string& name) { return (fs::temp_directory_path() / name).string(); }
void write_text(const std::string& path, const std::string& content) {
  std::ofstream o(path);
  o << content;
}
}  // namespace

TEST_CASE("loadtxt whitespace vs numpy") {
  const std::string p = tmp("numpp_loadtxt_ws.txt");
  write_text(p, "1 2 3\n4 5 6\n7 8 9\n");
  auto o = npt::oracle("a=np.loadtxt(r'" + p + "')");
  if (o) CHECK(allclose(loadtxt(p), *o, 1e-9, 1e-12, true));
}
TEST_CASE("loadtxt csv with comments vs numpy") {
  const std::string p = tmp("numpp_loadtxt_csv.txt");
  write_text(p, "# header\n1.5,2.5,3.5\n4.5,5.5,6.5\n");
  auto o = npt::oracle("a=np.loadtxt(r'" + p + "',delimiter=',')");
  if (o) CHECK(allclose(loadtxt(p, ','), *o, 1e-9, 1e-12, true));
}
TEST_CASE("loadtxt single column 1-D vs numpy") {
  const std::string p = tmp("numpp_loadtxt_col.txt");
  write_text(p, "1\n2\n3\n4\n");
  auto o = npt::oracle("a=np.loadtxt(r'" + p + "')");
  if (o) {
    ndarray r = loadtxt(p);
    CHECK(r.ndim() == 1);
    CHECK(allclose(r, *o, 1e-9, 1e-12, true));
  }
}
TEST_CASE("savetxt round-trips through numpy") {
  const std::string p = tmp("numpp_savetxt.txt");
  ndarray a = arange(0., 12., 1., kFloat64).reshape({3, 4});
  savetxt(p, a, "%.10e", " ");
  auto o = npt::oracle("a=np.loadtxt(r'" + p + "')");
  if (o) CHECK(allclose(a, *o, 1e-9, 1e-12, true));
}
TEST_CASE("genfromtxt fills missing with nan") {
  const std::string p = tmp("numpp_genfromtxt.txt");
  write_text(p, "1,2,3\n4,,6\n");
  ndarray r = genfromtxt(p, ',');
  CHECK(std::isnan(r.item<double>({1, 1})));
  CHECK(r.item<double>({1, 0}) == 4.0);
}
TEST_CASE("fromstring vs numpy") {
  auto o = npt::oracle("a=np.fromstring('1, 2, 3, 4, 5', sep=',')");
  if (o) CHECK(allclose(fromstring("1, 2, 3, 4, 5", ","), *o, 1e-9, 1e-12, true));
}
TEST_CASE("tofile/fromfile round-trip") {
  const std::string p = tmp("numpp_tofile.bin");
  ndarray a = arange(0., 10., 1., kFloat64);
  tofile(a, p);
  ndarray r = fromfile(p, kFloat64);
  CHECK(allclose(r, a, 1e-12, 1e-12, true));
}
TEST_CASE("fromfile matches numpy tofile") {
  const std::string p = tmp("numpp_np_tofile.bin");
  auto o = npt::oracle("x=np.arange(8.); x.tofile(r'" + p + "'); a=x");
  if (o) CHECK(allclose(fromfile(p, kFloat64), *o, 1e-12, 1e-12, true));
}

// ---- integer string formatting (compared to known numpy outputs) ----
TEST_CASE("binary_repr matches numpy") {
  CHECK(binary_repr(5) == "101");
  CHECK(binary_repr(6, 8) == "00000110");
  CHECK(binary_repr(-3, 8) == "11111101");
  CHECK(binary_repr(0) == "0");
}
TEST_CASE("base_repr matches numpy") {
  CHECK(base_repr(255, 16) == "FF");
  CHECK(base_repr(5, 2, 3) == "000101");
  CHECK(base_repr(-31, 16) == "-1F");
  CHECK(base_repr(0, 2) == "0");
}
