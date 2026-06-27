#include "numpp/numpp.hpp"
#include "numpp/core/limits.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <vector>

using namespace numpp;

namespace {
ndarray darr(const std::vector<double>& v) {
  ndarray a(Shape{static_cast<int64_t>(v.size())}, kFloat64, Order::C);
  for (size_t i = 0; i < v.size(); ++i) a.typed_data<double>()[i] = v[i];
  return a;
}
}  // namespace

TEST_CASE("finfo float64 attributes match numpy") {
  FInfo f = finfo(kFloat64);
  CHECK(f.bits == 64);
  CHECK(f.nmant == 52);
  CHECK(f.nexp == 11);
  CHECK(f.precision == 15);
  ndarray got = darr({f.eps, f.epsneg, f.tiny, f.max, f.resolution, f.smallest_subnormal});
  auto o = npt::oracle(
      "fi=np.finfo(np.float64); a=np.array([fi.eps,fi.epsneg,fi.tiny,fi.max,"
      "fi.resolution,fi.smallest_subnormal])");
  if (o) CHECK(allclose(got, *o, 1e-12, 0, true));
  CHECK(f.min == -f.max);
}

TEST_CASE("finfo float32 attributes match numpy") {
  FInfo f = finfo(kFloat32);
  CHECK(f.bits == 32);
  CHECK(f.nmant == 23);
  CHECK(f.nexp == 8);
  ndarray got = darr({f.eps, f.epsneg, f.tiny, f.max, f.resolution});
  auto o = npt::oracle(
      "fi=np.finfo(np.float32); a=np.array([fi.eps,fi.epsneg,fi.tiny,fi.max,fi.resolution],"
      "dtype=np.float64)");
  if (o) CHECK(allclose(got, *o, 1e-6, 0, true));
}

TEST_CASE("finfo float16 robust attributes match numpy") {
  FInfo f = finfo(kFloat16);
  CHECK(f.bits == 16);
  CHECK(f.nmant == 10);
  CHECK(f.nexp == 5);
  ndarray got = darr({f.eps, f.tiny, f.max, f.epsneg});
  auto o = npt::oracle(
      "fi=np.finfo(np.float16); a=np.array([fi.eps,fi.tiny,fi.max,fi.epsneg],dtype=np.float64)");
  if (o) CHECK(allclose(got, *o, 1e-6, 0, true));
}

TEST_CASE("finfo on complex reports the real component's limits") {
  CHECK(finfo(kComplex128).eps == finfo(kFloat64).eps);
  CHECK(finfo(kComplex64).max == finfo(kFloat32).max);
}

TEST_CASE("finfo rejects integer dtype") {
  CHECK_THROWS_AS(finfo(kInt32), value_error);
}

TEST_CASE("iinfo matches numpy for signed/unsigned widths") {
  CHECK(iinfo(kInt8).min == -128);
  CHECK(iinfo(kInt8).max == 127u);
  CHECK(iinfo(kInt32).min == -2147483648LL);
  CHECK(iinfo(kInt32).max == 2147483647ull);
  CHECK(iinfo(kInt64).min == (-9223372036854775807LL - 1));
  CHECK(iinfo(kInt64).max == 9223372036854775807ull);
  CHECK(iinfo(kUInt8).min == 0);
  CHECK(iinfo(kUInt8).max == 255u);
  CHECK(iinfo(kUInt64).max == 18446744073709551615ull);
  CHECK(iinfo(kInt16).bits == 16);
}

TEST_CASE("iinfo int32 bounds match numpy oracle") {
  auto o = npt::oracle("ii=np.iinfo(np.int32); a=np.array([ii.min,ii.max],dtype=np.int64)");
  if (!o) return;
  ndarray got(Shape{2}, kInt64, Order::C);
  got.typed_data<int64_t>()[0] = iinfo(kInt32).min;
  got.typed_data<int64_t>()[1] = static_cast<int64_t>(iinfo(kInt32).max);
  CHECK(allclose(got, *o, 0, 0, true));
}

TEST_CASE("iinfo rejects float dtype") {
  CHECK_THROWS_AS(iinfo(kFloat64), value_error);
}
