// Regression tests for #14: structured-dtype (kind 'V') .npy save/load. The NPY
// descr for a structured dtype is a Python list of (name, format) tuples; verify
// it round-trips in NumPP and interoperates with numpy both ways.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;
namespace fs = std::filesystem;

namespace {
std::string tmp(const std::string& name) { return (fs::temp_directory_path() / name).string(); }
}  // namespace

TEST_CASE("structured .npy in-memory round trip (#14)") {
  DType st = make_struct({{"a", kInt32}, {"b", kFloat64}, {"c", kInt8}});
  ndarray arr(Shape{3}, st, Order::C);
  std::memset(arr.bytes(), 0, static_cast<size_t>(arr.nbytes()));
  for (int i = 0; i < 3; ++i) {
    set_field<int32_t>(arr, i, "a", i + 1);
    set_field<double>(arr, i, "b", i * 1.5);
    set_field<int8_t>(arr, i, "c", static_cast<int8_t>(10 + i));
  }
  std::string bytes = npy_bytes(arr);
  ndarray b = npy_from_bytes(bytes.data(), bytes.size());
  CHECK(b.dtype().kind() == 'V');
  CHECK(b.dtype() == st);
  CHECK((b.shape() == Shape{3}));
  for (int i = 0; i < 3; ++i) {
    CHECK(get_field<int32_t>(b, i, "a") == i + 1);
    CHECK(get_field<double>(b, i, "b") == i * 1.5);
    CHECK(get_field<int8_t>(b, i, "c") == static_cast<int8_t>(10 + i));
  }
}

TEST_CASE("structured .npy saved by NumPP reads in numpy (#14)") {
  if (!npt::numpy_available()) return;
  DType st = make_struct({{"a", kInt32}, {"b", kFloat64}});
  ndarray arr(Shape{2}, st, Order::C);
  std::memset(arr.bytes(), 0, static_cast<size_t>(arr.nbytes()));
  set_field<int32_t>(arr, 0, "a", 42); set_field<double>(arr, 0, "b", 2.5);
  set_field<int32_t>(arr, 1, "a", 7);  set_field<double>(arr, 1, "b", -3.0);
  std::string path = tmp("numpp_struct.npy");
  save(path, arr);
  auto oa = npt::oracle("d=np.load(r'" + path + "'); a=d['a']");
  if (oa) { CHECK(oa->item<int32_t>({0}) == 42); CHECK(oa->item<int32_t>({1}) == 7); }
  auto ob = npt::oracle("d=np.load(r'" + path + "'); a=d['b']");
  if (ob) { CHECK(ob->item<double>({0}) == 2.5); CHECK(ob->item<double>({1}) == -3.0); }
}

TEST_CASE("structured .npy written by numpy loads in NumPP (#14)") {
  if (!npt::numpy_available()) return;
  std::string path = tmp("np_struct.npy");
  std::string cmd =
      "python3 -c \"import numpy as np; x=np.zeros(2,dtype=[('a','<i4'),('b','<f8')]); "
      "x['a']=[5,6]; x['b']=[1.25,9.5]; np.save(r'" + path + "', x)\"";
  CHECK(std::system(cmd.c_str()) == 0);
  ndarray a = load(path);
  CHECK(a.dtype().kind() == 'V');
  CHECK((a.shape() == Shape{2}));
  CHECK(get_field<int32_t>(a, 0, "a") == 5);
  CHECK(get_field<int32_t>(a, 1, "a") == 6);
  CHECK(get_field<double>(a, 0, "b") == 1.25);
  CHECK(get_field<double>(a, 1, "b") == 9.5);
}
