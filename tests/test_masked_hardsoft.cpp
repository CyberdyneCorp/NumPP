#include "numpp/numpp.hpp"
#include "numpp/ma/masked.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

namespace {
// [1., 2., 3.] with element 1 masked.
ma::MaskedArray make_m() {
  ndarray data = arange(1.0, 4.0, 1.0);          // [1,2,3] float64
  ndarray mask = zeros(Shape{3}, kBool);
  mask.set_item<bool>({1}, true);
  return ma::masked_array(data, mask);
}
}  // namespace

TEST_CASE("default mask is soft") {
  ma::MaskedArray m = make_m();
  CHECK(ma::hardmask(m) == false);
}

TEST_CASE("soft mask: assignment writes value and clears the mask vs numpy") {
  ma::MaskedArray m = make_m();
  ma::setitem(m, {1}, 9.0);                       // soft: writes + unmasks
  CHECK(ma::hardmask(m) == false);
  CHECK(ma::count(m) == 3);                        // now nothing masked
  ndarray got = ma::filled(m, -1.0);
  auto o = npt::oracle("m=np.ma.array([1.,2.,3.],mask=[False,True,False]); "
                       "m[1]=9.; a=m.filled(-1.)");
  if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
}

TEST_CASE("hard mask: assignment through a masked cell is a no-op vs numpy") {
  ma::MaskedArray m = make_m();
  ma::harden_mask(m);
  CHECK(ma::hardmask(m) == true);
  ma::setitem(m, {1}, 9.0);                       // hard + masked: ignored
  CHECK(ma::count(m) == 2);                        // still masked
  ndarray got = ma::filled(m, -1.0);
  auto o = npt::oracle("m=np.ma.array([1.,2.,3.],mask=[False,True,False]); "
                       "m.harden_mask(); m[1]=9.; a=m.filled(-1.)");
  if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
}

TEST_CASE("hard mask still allows writing an unmasked cell vs numpy") {
  ma::MaskedArray m = make_m();
  ma::harden_mask(m);
  ma::setitem(m, {0}, 7.0);                       // index 0 is unmasked -> writes
  ndarray got = ma::filled(m, -1.0);
  auto o = npt::oracle("m=np.ma.array([1.,2.,3.],mask=[False,True,False]); "
                       "m.harden_mask(); m[0]=7.; a=m.filled(-1.)");
  if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
}

TEST_CASE("soften_mask re-enables assignment through a masked cell vs numpy") {
  ma::MaskedArray m = make_m();
  ma::harden_mask(m);
  ma::soften_mask(m);
  CHECK(ma::hardmask(m) == false);
  ma::setitem(m, {1}, 9.0);                       // soft again: writes + unmasks
  ndarray got = ma::filled(m, -1.0);
  auto o = npt::oracle("m=np.ma.array([1.,2.,3.],mask=[False,True,False]); "
                       "m.harden_mask(); m.soften_mask(); m[1]=9.; a=m.filled(-1.)");
  if (o) CHECK(allclose(got, *o, 1e-12, 1e-12, true));
}
