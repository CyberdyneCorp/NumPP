#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("fix vs numpy") {
  ndarray x = zeros({4}, kFloat64);
  x.set_item<double>({0}, -1.5);
  x.set_item<double>({1}, 1.5);
  x.set_item<double>({2}, 2.7);
  x.set_item<double>({3}, -0.4);
  auto o = npt::oracle("a=np.fix([-1.5,1.5,2.7,-0.4])");
  if (o) CHECK(allclose(fix(x), *o, 1e-12, 1e-12, true));
}

TEST_CASE("real_if_close vs numpy") {
  ndarray c = zeros({2}, kComplex128);
  c.set_item<std::complex<double>>({0}, {1.0, 1e-15});
  c.set_item<std::complex<double>>({1}, {2.0, 0.0});
  auto o = npt::oracle("a=np.real_if_close([1+1e-15j, 2+0j])");
  if (o) CHECK(allclose(real_if_close(c), *o, 1e-12, 1e-12, true));

  // Imag too large -> array returned unchanged (still complex).
  ndarray big = zeros({2}, kComplex128);
  big.set_item<std::complex<double>>({0}, {1.0, 0.5});
  big.set_item<std::complex<double>>({1}, {2.0, 0.0});
  CHECK(iscomplexobj(real_if_close(big)));
}

TEST_CASE("iscomplex / isreal vs numpy") {
  ndarray c = zeros({3}, kComplex128);
  c.set_item<std::complex<double>>({0}, {1.0, 1.0});
  c.set_item<std::complex<double>>({1}, {2.0, 0.0});
  c.set_item<std::complex<double>>({2}, {0.0, 3.0});
  auto oc = npt::oracle("a=np.iscomplex([1+1j, 2+0j, 3j])");
  if (oc) CHECK(allclose(iscomplex(c), *oc, 1e-12, 1e-12, true));
  auto orr = npt::oracle("a=np.isreal([1+1j, 2+0j, 3j])");
  if (orr) CHECK(allclose(isreal(c), *orr, 1e-12, 1e-12, true));

  // A real-dtype array: nothing is complex.
  ndarray r = arange(0., 4., 1., kFloat64);
  auto orealobj = npt::oracle("a=np.iscomplex([0.,1.,2.,3.])");
  if (orealobj) CHECK(allclose(iscomplex(r), *orealobj, 1e-12, 1e-12, true));
}

TEST_CASE("iscomplexobj / isrealobj / common_type") {
  ndarray r = arange(0., 3., 1., kFloat64);
  ndarray c = zeros({2}, kComplex128);
  CHECK(iscomplexobj(c));
  CHECK(!(iscomplexobj(r)));
  CHECK(isrealobj(r));
  CHECK(!(isrealobj(c)));

  CHECK((common_type({r, arange(0, 3, 1, kInt64)}).id() == kFloat64.id()));
  CHECK((common_type({r, c}).id() == kComplex128.id()));
  CHECK((common_type({arange(0, 3, 1, kInt32)}).id() == kFloat64.id()));
}

TEST_CASE("packbits vs numpy") {
  ndarray bits = zeros({8}, kUInt8);
  int pattern[8] = {1, 0, 1, 1, 0, 0, 0, 1};
  for (int i = 0; i < 8; ++i) bits.set_item<uint8_t>({i}, static_cast<uint8_t>(pattern[i]));
  auto o = npt::oracle("a=np.packbits([1,0,1,1,0,0,0,1])");
  if (o) CHECK(allclose(packbits(bits), *o, 1e-12, 1e-12, true));

  // Non-multiple-of-8 length: padded with zeros.
  ndarray b3 = zeros({3}, kUInt8);
  b3.set_item<uint8_t>({0}, 1);
  b3.set_item<uint8_t>({2}, 1);
  auto o2 = npt::oracle("a=np.packbits([1,0,1])");
  if (o2) CHECK(allclose(packbits(b3), *o2, 1e-12, 1e-12, true));
}

TEST_CASE("unpackbits vs numpy") {
  ndarray u = zeros({1}, kUInt8);
  u.set_item<uint8_t>({0}, static_cast<uint8_t>(0xb1));
  auto o = npt::oracle("a=np.unpackbits(np.array([0xb1], dtype=np.uint8))");
  if (o) CHECK(allclose(unpackbits(u), *o, 1e-12, 1e-12, true));

  ndarray u2 = zeros({2}, kUInt8);
  u2.set_item<uint8_t>({0}, static_cast<uint8_t>(0x0f));
  u2.set_item<uint8_t>({1}, static_cast<uint8_t>(0xa0));
  auto o2 = npt::oracle("a=np.unpackbits(np.array([0x0f,0xa0], dtype=np.uint8))");
  if (o2) CHECK(allclose(unpackbits(u2), *o2, 1e-12, 1e-12, true));
}

TEST_CASE("emath::sqrt vs numpy") {
  ndarray neg = zeros({2}, kFloat64);
  neg.set_item<double>({0}, -4.0);
  neg.set_item<double>({1}, 4.0);
  auto o = npt::oracle("a=np.emath.sqrt([-4., 4.])");
  if (o) CHECK(allclose(emath::sqrt(neg), *o, 1e-9, 1e-12, true));

  // All non-negative -> real result.
  ndarray pos = arange(0., 5., 1., kFloat64);
  auto op = npt::oracle("a=np.emath.sqrt([0.,1.,2.,3.,4.])");
  if (op) CHECK(allclose(emath::sqrt(pos), *op, 1e-9, 1e-12, true));
}

TEST_CASE("emath::log vs numpy") {
  ndarray neg = zeros({2}, kFloat64);
  neg.set_item<double>({0}, -1.0);
  neg.set_item<double>({1}, 1.0);
  auto o = npt::oracle("a=np.emath.log([-1., 1.])");
  if (o) CHECK(allclose(emath::log(neg), *o, 1e-9, 1e-12, true));

  ndarray pos = arange(1., 5., 1., kFloat64);
  auto op = npt::oracle("a=np.emath.log([1.,2.,3.,4.])");
  if (op) CHECK(allclose(emath::log(pos), *op, 1e-9, 1e-12, true));
}

TEST_CASE("emath::power vs numpy") {
  ndarray base = zeros({2}, kFloat64);
  base.set_item<double>({0}, 2.0);
  base.set_item<double>({1}, 3.0);
  ndarray exp2 = full({1}, 2.0, kFloat64);
  auto o = npt::oracle("a=np.emath.power([2.,3.], 2.)");
  if (o) CHECK(allclose(emath::power(base, exp2), *o, 1e-9, 1e-12, true));

  // Negative base + fractional exponent -> complex.
  ndarray nbase = full({1}, -2.0, kFloat64);
  ndarray half = full({1}, 0.5, kFloat64);
  auto oc = npt::oracle("a=np.emath.power([-2.], 0.5)");
  if (oc) CHECK(allclose(emath::power(nbase, half), *oc, 1e-9, 1e-12, true));
}
