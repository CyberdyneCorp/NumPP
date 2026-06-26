#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("poly1d eval vs numpy") {
  ndarray c({4}, kFloat64, Order::C);
  c.set_item<double>({0}, 1.0);
  c.set_item<double>({1}, -2.0);
  c.set_item<double>({2}, 0.0);
  c.set_item<double>({3}, 3.0);
  poly1d p(c);

  ndarray x({4}, kFloat64, Order::C);
  x.set_item<double>({0}, -1.0);
  x.set_item<double>({1}, 0.0);
  x.set_item<double>({2}, 1.0);
  x.set_item<double>({3}, 2.0);

  auto o = npt::oracle("p=np.poly1d([1,-2,0,3]); a=p(np.array([-1,0,1,2.]))");
  if (o) CHECK(allclose(p(x), *o, 1e-9, 1e-12, true));

  auto oo = npt::oracle("a=np.array([3.])");
  if (oo) CHECK(p.order() == 3);
}

TEST_CASE("poly1d coeffs trimming vs numpy") {
  ndarray c({5}, kFloat64, Order::C);
  c.set_item<double>({0}, 0.0);
  c.set_item<double>({1}, 0.0);
  c.set_item<double>({2}, 1.0);
  c.set_item<double>({3}, 2.0);
  c.set_item<double>({4}, 3.0);
  poly1d p(c);
  auto o = npt::oracle("a=np.poly1d([0,0,1,2,3]).coeffs");
  if (o) CHECK(allclose(p.coeffs(), *o, 1e-9, 1e-12, true));
  CHECK(p.order() == 2);
}

TEST_CASE("poly1d deriv and integ vs numpy") {
  ndarray c({4}, kFloat64, Order::C);
  c.set_item<double>({0}, 1.0);
  c.set_item<double>({1}, -2.0);
  c.set_item<double>({2}, 0.0);
  c.set_item<double>({3}, 3.0);
  poly1d p(c);

  auto od = npt::oracle("a=np.poly1d([1,-2,0,3]).deriv().coeffs");
  if (od) CHECK(allclose(p.deriv().coeffs(), *od, 1e-9, 1e-12, true));

  auto od2 = npt::oracle("a=np.poly1d([1,-2,0,3]).deriv(2).coeffs");
  if (od2) CHECK(allclose(p.deriv(2).coeffs(), *od2, 1e-9, 1e-12, true));

  auto oi = npt::oracle("a=np.poly1d([1,-2,0,3]).integ().coeffs");
  if (oi) CHECK(allclose(p.integ().coeffs(), *oi, 1e-9, 1e-12, true));

  auto oik = npt::oracle("a=np.poly1d([1,-2,0,3]).integ(1,5).coeffs");
  if (oik) CHECK(allclose(p.integ(1, 5.0).coeffs(), *oik, 1e-9, 1e-12, true));
}

TEST_CASE("poly1d roots vs numpy") {
  ndarray c({3}, kFloat64, Order::C);
  c.set_item<double>({0}, 1.0);
  c.set_item<double>({1}, -3.0);
  c.set_item<double>({2}, 2.0);
  poly1d p(c);
  // sorted real parts of roots
  ndarray r = real(p.roots()).astype(kFloat64);
  ndarray rs = sort(r);
  auto o = npt::oracle("a=np.sort(np.poly1d([1,-3,2]).roots.real)");
  if (o) CHECK(allclose(rs, *o, 1e-7, 1e-9, true));
}

TEST_CASE("poly1d arithmetic vs numpy") {
  ndarray a({2}, kFloat64, Order::C);
  a.set_item<double>({0}, 1.0);
  a.set_item<double>({1}, 2.0);
  ndarray b({3}, kFloat64, Order::C);
  b.set_item<double>({0}, 1.0);
  b.set_item<double>({1}, 0.0);
  b.set_item<double>({2}, 3.0);
  poly1d pa(a), pb(b);

  auto om = npt::oracle("a=(np.poly1d([1,2])*np.poly1d([1,0,3])).coeffs");
  if (om) CHECK(allclose((pa * pb).coeffs(), *om, 1e-9, 1e-12, true));

  auto os = npt::oracle("a=(np.poly1d([1,0,3])+np.poly1d([1,2])).coeffs");
  if (os) CHECK(allclose((pb + pa).coeffs(), *os, 1e-9, 1e-12, true));

  auto od = npt::oracle("a=(np.poly1d([1,0,3])-np.poly1d([1,2])).coeffs");
  if (od) CHECK(allclose((pb - pa).coeffs(), *od, 1e-9, 1e-12, true));
}

TEST_CASE("polyfit_weighted vs numpy") {
  ndarray x = arange(0., 6., 1., kFloat64);
  ndarray y({6}, kFloat64, Order::C);
  y.set_item<double>({0}, 1.0);
  y.set_item<double>({1}, 0.5);
  y.set_item<double>({2}, 3.0);
  y.set_item<double>({3}, 2.5);
  y.set_item<double>({4}, 6.0);
  y.set_item<double>({5}, 9.0);
  ndarray w({6}, kFloat64, Order::C);
  for (int64_t i = 0; i < 6; ++i) w.set_item<double>({i}, 1.0 + 0.1 * i);

  auto o = npt::oracle(
      "x=np.arange(0.,6.,1.); y=np.array([1,0.5,3,2.5,6,9.]); "
      "w=np.array([1.0,1.1,1.2,1.3,1.4,1.5]); a=np.polyfit(x,y,2,w=w)");
  if (o) CHECK(allclose(polyfit_weighted(x, y, 2, w), *o, 1e-6, 1e-8, true));
}
