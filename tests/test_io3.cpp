#include <cstdio>
#include <string>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

// Capture python stdout for str()/repr() comparison.
static std::string py_str(const std::string& a) { std::string cmd = "import numpy as np,sys; a=" + a + "; sys.stdout.write(str(a))"; std::string out; if (FILE* f = popen(("python3 -c \"" + cmd + "\"").c_str(), "r")) { char b[8192]; size_t n; while ((n = fread(b, 1, sizeof(b), f)) > 0) out.append(b, n); pclose(f); } return out; }
static std::string py_repr(const std::string& a) { std::string cmd = "import numpy as np,sys; a=" + a + "; sys.stdout.write(repr(a))"; std::string out; if (FILE* f = popen(("python3 -c \"" + cmd + "\"").c_str(), "r")) { char b[8192]; size_t n; while ((n = fread(b, 1, sizeof(b), f)) > 0) out.append(b, n); pclose(f); } return out; }

static void chk_str(const ndarray& got, const std::string& npexpr, const char* label) {
  if (!npt::numpy_available()) { std::fprintf(stderr, "  [skip] %s\n", label); return; }
  std::string want = py_str(npexpr);
  if (array_str(got) != want)
    std::fprintf(stderr, "  STR mismatch %s:\n  got : [%s]\n  want: [%s]\n", label, array_str(got).c_str(), want.c_str());
  CHECK(array_str(got) == want);
}
static void chk_repr(const ndarray& got, const std::string& npexpr, const char* label) {
  if (!npt::numpy_available()) { std::fprintf(stderr, "  [skip] %s\n", label); return; }
  std::string want = py_repr(npexpr);
  if (array_repr(got) != want)
    std::fprintf(stderr, "  REPR mismatch %s:\n  got : [%s]\n  want: [%s]\n", label, array_repr(got).c_str(), want.c_str());
  CHECK(array_repr(got) == want);
}

TEST_CASE("str/repr: 1d int") {
  ndarray a = arange(1.0, 4.0, 1.0).astype(kInt64);
  chk_str(a, "np.arange(1,4)", "1d-int-str");
  chk_repr(a, "np.arange(1,4)", "1d-int-repr");
}
TEST_CASE("str/repr: 1d float") {
  ndarray a({3}, kFloat64); a.set_item<double>({0}, -1.5); a.set_item<double>({1}, 2.0); a.set_item<double>({2}, 30.25);
  chk_str(a, "np.array([-1.5,2.,30.25])", "1d-float-str");
  chk_repr(a, "np.array([-1.5,2.,30.25])", "1d-float-repr");
}
TEST_CASE("str/repr: 2d") {
  ndarray a = arange(1.0, 7.0, 1.0).reshape({2, 3}).astype(kInt64);
  chk_str(a, "np.arange(1,7).reshape(2,3)", "2d-int-str");
  chk_repr(a, "np.arange(1,7).reshape(2,3)", "2d-int-repr");
  ndarray f = arange(1.0, 5.0, 1.0).astype(kFloat64).reshape({2, 2});
  chk_str(f, "np.arange(1,5.).reshape(2,2)", "2d-float-str");
}
TEST_CASE("str/repr: bool and dtype suffix") {
  ndarray b({3}, kBool); b.set_item<bool>({0}, true); b.set_item<bool>({1}, false); b.set_item<bool>({2}, true);
  chk_str(b, "np.array([True,False,True])", "bool-str");
  chk_repr(b, "np.array([True,False,True])", "bool-repr");
  chk_repr(arange(1.0, 4.0, 1.0).astype(kInt32), "np.arange(1,4,dtype=np.int32)", "int32-repr");
  chk_repr(arange(1.0, 3.0, 1.0).astype(kFloat32), "np.array([1.,2.],dtype=np.float32)", "float32-repr");
}
TEST_CASE("str/repr: 3d int") {
  ndarray a = arange(0.0, 8.0, 1.0).reshape({2, 2, 2}).astype(kInt64);
  chk_str(a, "np.arange(8).reshape(2,2,2)", "3d-str");
  chk_repr(a, "np.arange(8).reshape(2,2,2)", "3d-repr");
}
TEST_CASE("str/repr: summarized 1d (>1000)") {
  ndarray a = arange(0.0, 2000.0, 1.0).astype(kInt64);
  chk_str(a, "np.arange(2000)", "summ-str");
}
