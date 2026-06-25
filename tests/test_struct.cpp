#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static std::string pystr(const std::string& code, bool rep) {
  std::string out, fn = rep ? "repr" : "str";
  std::string cmd = "import numpy as np,sys; a=" + code + "; sys.stdout.write(" + fn + "(a))";
  if (FILE* f = popen(("python3 -c \"" + cmd + "\"").c_str(), "r")) { char b[4096]; size_t n; while ((n = fread(b, 1, sizeof(b), f)) > 0) out.append(b, n); pclose(f); }
  return out;
}

TEST_CASE("structured dtype layout") {
  DType st = make_struct({{"a", kInt32}, {"b", kFloat64}, {"c", kInt8}});
  CHECK(st.kind() == 'V');
  CHECK(st.itemsize() == 13);  // 4 + 8 + 1, no padding
  CHECK(st.meta()->fields[1].name == "b");
  CHECK(st.meta()->fields[1].offset == 4);
  CHECK(st.meta()->fields[2].offset == 12);
}

TEST_CASE("structured array field views alias the parent") {
  DType st = make_struct({{"a", kInt32}, {"b", kFloat64}});
  ndarray arr(Shape{3}, st, Order::C);
  std::memset(arr.bytes(), 0, static_cast<size_t>(arr.nbytes()));
  set_field<int32_t>(arr, 0, "a", 1); set_field<double>(arr, 0, "b", 2.5);
  set_field<int32_t>(arr, 1, "a", 7); set_field<double>(arr, 1, "b", -3.0);
  set_field<int32_t>(arr, 2, "a", 9); set_field<double>(arr, 2, "b", 100.25);
  CHECK(get_field<int32_t>(arr, 1, "a") == 7);
  CHECK(get_field<double>(arr, 2, "b") == 100.25);
  // field_view is a real view: writing through it changes the record
  ndarray va = field_view(arr, "a");
  CHECK(va.dtype() == kInt32 && va.shape() == Shape({3}));
  va.set_item<int32_t>({0}, 42);
  CHECK(get_field<int32_t>(arr, 0, "a") == 42);
  // field 'a' values equal numpy's
  auto o = npt::oracle("x=np.zeros(3,dtype=[('a','i4'),('b','f8')]);x['a']=[42,7,9];a=x['a']");
  if (o) for (int i = 0; i < 3; ++i) CHECK(va.item<int32_t>({i}) == o->item<int32_t>({i}));
}

TEST_CASE("structured str/repr (best-effort, issue #12-style if diverges)") {
  DType st = make_struct({{"a", kInt32}, {"b", kFloat64}});
  ndarray arr(Shape{2}, st, Order::C);
  std::memset(arr.bytes(), 0, static_cast<size_t>(arr.nbytes()));
  set_field<int32_t>(arr, 0, "a", 1); set_field<double>(arr, 0, "b", 2.0);
  set_field<int32_t>(arr, 1, "a", 3); set_field<double>(arr, 1, "b", 4.0);
  std::string s = array_str(arr), r = array_repr(arr);
  std::fprintf(stderr, "  struct str : %s\n  struct repr: %s\n", s.c_str(), r.c_str());
  if (npt::numpy_available()) {
    std::string code = "np.array([(1,2.0),(3,4.0)],dtype=[('a','i4'),('b','f8')])";
    // pinned-current behavior; matches numpy where feasible
    CHECK(s == pystr(code, false));
    CHECK(r == pystr(code, true));
  }
}
