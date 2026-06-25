#include <cstdio>
#include <filesystem>
#include <string>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;
namespace fs = std::filesystem;

static std::string tmp(const std::string& name) { return (fs::temp_directory_path() / name).string(); }

TEST_CASE("npy round trip in-memory (all dtypes, shapes)") {
  std::vector<DType> dts = {kBool, kInt8, kInt32, kInt64, kUInt16, kFloat32, kFloat64, kComplex128};
  for (DType dt : dts) {
    ndarray a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4}).astype(dt);
    std::string bytes = npy_bytes(a);
    ndarray b = npy_from_bytes(bytes.data(), bytes.size());
    CHECK(b.dtype() == dt);
    CHECK(b.shape() == a.shape());
    CHECK(array_equal(a, b));
  }
  // 1-D and 0-D shape strings
  ndarray v = arange(0.0, 5.0, 1.0);
  CHECK(array_equal(v, npy_from_bytes(npy_bytes(v).data(), npy_bytes(v).size())));
  ndarray s = zeros({}, kFloat64); s.set_item<double>({}, 7.0);
  CHECK(array_equal(s, npy_from_bytes(npy_bytes(s).data(), npy_bytes(s).size())));
}

TEST_CASE("save then load from disk") {
  ndarray a = arange(1.0, 13.0, 1.0).reshape({3, 4}).astype(kFloat32);
  std::string path = tmp("numpp_io_test.npy");
  save(path, a);
  ndarray b = load(path);
  CHECK(b.dtype() == kFloat32);
  CHECK(array_equal(a, b));
  std::error_code ec; fs::remove(path, ec);
}

TEST_CASE("numpp save -> numpy load matches (oracle)") {
  if (!npt::numpy_available()) { std::fprintf(stderr, "  [skip] np load\n"); return; }
  ndarray a = arange(0.0, 12.0, 1.0).reshape({3, 4}).astype(kInt32);
  std::string path = tmp("numpp_to_np.npy");
  save(path, a);
  // numpy reads our file and re-saves; we reload and compare.
  std::string code = "a=np.load(r'" + path + "')";
  auto o = npt::oracle(code);
  std::error_code ec; fs::remove(path, ec);
  if (!o) { std::fprintf(stderr, "  [skip] np load\n"); return; }
  CHECK(array_equal(a, *o));
}

TEST_CASE("numpy save -> numpp load matches (fortran_order too)") {
  if (!npt::numpy_available()) { std::fprintf(stderr, "  [skip] np save\n"); return; }
  std::string cpath = tmp("np_c.npy"), fpath = tmp("np_f.npy");
  std::string py = "import numpy as np\n"
                   "np.save(r'" + cpath + "', np.arange(12,dtype=np.float64).reshape(3,4))\n"
                   "np.save(r'" + fpath + "', np.asfortranarray(np.arange(12,dtype=np.float64).reshape(3,4)))\n";
  std::string pyfile = tmp("np_io.py");
  { std::ofstream s(pyfile); s << py; }
  if (std::system(("python3 " + pyfile + " >/dev/null 2>&1").c_str()) != 0) { std::fprintf(stderr, "  [skip]\n"); return; }
  ndarray refc = arange(0.0, 12.0, 1.0).reshape({3, 4});
  ndarray c = load(cpath), f = load(fpath);
  CHECK(array_equal(c, refc));
  CHECK(array_equal(f, refc));   // fortran_order file loads with correct values
  CHECK(f.f_contiguous());
  std::error_code ec; fs::remove(cpath, ec); fs::remove(fpath, ec); fs::remove(pyfile, ec);
}
