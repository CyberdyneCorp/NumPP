#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

TEST_CASE("dtype metadata and names") {
  CHECK(kFloat64.itemsize() == 8);
  CHECK(kComplex128.itemsize() == 16);
  CHECK(kInt8.itemsize() == 1);
  CHECK(kFloat16.itemsize() == 2);
  CHECK(kUInt16.kind() == 'u');
  CHECK(kComplex64.kind() == 'c');
  CHECK(std::string(kInt32.name()) == "int32");
  CHECK(DType::from_name("float32") == kFloat32);
  CHECK(DType::from_name("i4") == kInt32);
  CHECK(DType::from_name("c16") == kComplex128);
  CHECK(DType::from_name("float") == kFloat64);
  CHECK_THROWS_AS(DType::from_name("float7"), type_error);
  CHECK(kFloat64.byteorder() == '=');
}

TEST_CASE("default dtypes") {
  CHECK(zeros({2}).dtype() == kFloat64);
  CHECK(arange(3.0).dtype() == default_int());
  CHECK(default_int() == kInt64);  // LP64 platform
}

TEST_CASE("result_type matches NumPy for all dtype pairs") {
  const std::vector<DType> all = {kBool, kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16,
                                  kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kComplex64,
                                  kComplex128};
  // Build the 14x14 promotion table from NumPy as type-index codes.
  auto o = npt::oracle(
      "order=['bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64',"
      "'float16','float32','float64','complex64','complex128']\n"
      "idx={n:i for i,n in enumerate(order)}\n"
      "a=np.array([[idx[np.result_type(np.dtype(x),np.dtype(y)).name] for y in order] for x in order],dtype=np.int8)");
  if (!o) { std::fprintf(stderr, "  [skip] result_type all-pairs (no numpy)\n"); return; }
  for (int i = 0; i < 14; ++i)
    for (int j = 0; j < 14; ++j)
      CHECK(static_cast<int>(result_type(all[i], all[j]).id()) ==
            static_cast<int>(o->item<int8_t>({i, j})));
}

TEST_CASE("can_cast modes") {
  CHECK(can_cast(kInt16, kInt32, Casting::Safe));
  CHECK(!can_cast(kFloat64, kInt32, Casting::Safe));
  CHECK(can_cast(kFloat64, kFloat32, Casting::SameKind));
  CHECK(!can_cast(kFloat64, kFloat32, Casting::Safe));
  CHECK(can_cast(kInt32, kInt8, Casting::Unsafe));
  CHECK(can_cast(kFloat32, kFloat32, Casting::No));
  CHECK(!can_cast(kFloat32, kFloat64, Casting::No));
}

TEST_CASE("astype conversions") {
  // float -> int truncates toward zero
  ndarray f({3}, kFloat64);
  f.set_item<double>({0}, 1.9);
  f.set_item<double>({1}, -1.9);
  f.set_item<double>({2}, 2.5);
  ndarray i = f.astype(kInt32);
  CHECK(i.item<int32_t>({0}) == 1);
  CHECK(i.item<int32_t>({1}) == -1);
  CHECK(i.item<int32_t>({2}) == 2);

  // real -> complex
  ndarray c = f.astype(kComplex128);
  CHECK(c.item<std::complex<double>>({0}) == std::complex<double>(1.9, 0));

  // half round-trip
  ndarray h = f.astype(kFloat16).astype(kFloat64);
  CHECK(std::abs(h.item<double>({2}) - 2.5) < 1e-3);
}
