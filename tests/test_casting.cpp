#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

using namespace numpp;

namespace {
// Encode a result dtype as numpy does for comparison: [ord(kind), itemsize].
ndarray encode(const std::vector<DType>& ds) {
  ndarray a(Shape{static_cast<int64_t>(ds.size()), 2}, kInt64, Order::C);
  for (size_t i = 0; i < ds.size(); ++i) {
    a.typed_data<int64_t>()[2 * i] = static_cast<int64_t>(static_cast<unsigned char>(ds[i].kind()));
    a.typed_data<int64_t>()[2 * i + 1] = ds[i].itemsize();
  }
  return a;
}
}  // namespace

TEST_CASE("promote_types matches numpy across a dtype matrix") {
  std::vector<std::pair<DType, DType>> P = {
      {kInt8, kUInt8},   {kInt16, kUInt16},      {kInt32, kInt64},
      {kFloat32, kInt32},{kFloat16, kInt16},     {kFloat32, kFloat64},
      {kComplex64, kFloat64}, {kBool, kInt8},    {kUInt32, kInt8},
      {kInt64, kUInt64}, {kFloat16, kFloat16},   {kComplex64, kComplex128},
      {kUInt8, kUInt8},  {kFloat64, kInt64}};
  std::vector<DType> got;
  for (auto& pr : P) got.push_back(promote_types(pr.first, pr.second));

  auto o = npt::oracle(
      "P=[('int8','uint8'),('int16','uint16'),('int32','int64'),('float32','int32'),"
      "('float16','int16'),('float32','float64'),('complex64','float64'),('bool','int8'),"
      "('uint32','int8'),('int64','uint64'),('float16','float16'),('complex64','complex128'),"
      "('uint8','uint8'),('float64','int64')];"
      "a=np.array([[ord(np.promote_types(x,y).kind),np.promote_types(x,y).itemsize] for x,y in P])");
  if (o) CHECK(allclose(encode(got), *o, 0, 0, true));
}

TEST_CASE("min_scalar_type matches numpy for integers") {
  std::vector<long long> IV = {0, 127, 128, 255, 256, 65535, 65536, -1, -128,
                               -129, -32768, -32769, 2147483647, 2147483648LL,
                               -2147483648LL, -2147483649LL};
  std::vector<DType> got;
  for (long long v : IV) got.push_back(min_scalar_type(v));
  auto o = npt::oracle(
      "IV=[0,127,128,255,256,65535,65536,-1,-128,-129,-32768,-32769,2147483647,"
      "2147483648,-2147483648,-2147483649];"
      "a=np.array([[ord(np.min_scalar_type(v).kind),np.min_scalar_type(v).itemsize] for v in IV])");
  if (o) CHECK(allclose(encode(got), *o, 0, 0, true));
}

TEST_CASE("min_scalar_type matches numpy for floats") {
  std::vector<double> FV = {0.0, 3.0, 3.14, 65504.0, 65505.0, 70000.0,
                            3.0e38, 3.5e38, 1e39, 1e300};
  std::vector<DType> got;
  for (double v : FV) got.push_back(min_scalar_type(v));
  auto o = npt::oracle(
      "FV=[0.0,3.0,3.14,65504.0,65505.0,70000.0,3.0e38,3.5e38,1e39,1e300];"
      "a=np.array([[ord(np.min_scalar_type(v).kind),np.min_scalar_type(v).itemsize] for v in FV])");
  if (o) CHECK(allclose(encode(got), *o, 0, 0, true));
}

TEST_CASE("min_scalar_type non-finite maps to float16") {
  CHECK(min_scalar_type(std::numeric_limits<double>::infinity()) == kFloat16);
  CHECK(min_scalar_type(std::numeric_limits<double>::quiet_NaN()) == kFloat16);
}

TEST_CASE("promote_types rejects extended dtypes") {
  CHECK_THROWS_AS(promote_types(make_string(3), kInt32), type_error);
}
