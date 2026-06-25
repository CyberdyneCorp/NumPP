#include <cmath>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s\n", label); return; }
  CHECK(allclose(got, *o, 1e-9, 1e-12, true));
}
static ndarray withnan() {
  ndarray a({2, 3}, kFloat64);
  double v[6] = {1.0, NAN, 3.0, 4.0, 5.0, NAN};
  for (int i = 0; i < 6; ++i) a.set_item<double>({i / 3, i % 3}, v[i]);
  return a;
}
static const char* W = "W=np.array([1.,np.nan,3,4,5,np.nan]).reshape(2,3)";

TEST_CASE("nansum/nanmean vs numpy") {
  chk(nansum(withnan()), std::string(W) + ";a=np.array(np.nansum(W))", "nansum all");
  chk(nansum(withnan(), 0), std::string(W) + ";a=np.nansum(W,axis=0)", "nansum axis0");
  chk(nanmean(withnan()), std::string(W) + ";a=np.array(np.nanmean(W))", "nanmean all");
  chk(nanmean(withnan(), 1), std::string(W) + ";a=np.nanmean(W,axis=1)", "nanmean axis1");
}
TEST_CASE("nanmin/nanmax vs numpy") {
  chk(nanmin(withnan()), std::string(W) + ";a=np.array(np.nanmin(W))", "nanmin");
  chk(nanmax(withnan(), 0), std::string(W) + ";a=np.nanmax(W,axis=0)", "nanmax axis0");
}
TEST_CASE("nanvar/nanstd vs numpy") {
  chk(nanvar(withnan()), std::string(W) + ";a=np.array(np.nanvar(W))", "nanvar");
  chk(nanstd(withnan(), 1), std::string(W) + ";a=np.nanstd(W,axis=1)", "nanstd axis1");
  chk(nanvar(withnan(), std::nullopt, false, 1), std::string(W) + ";a=np.array(np.nanvar(W,ddof=1))", "nanvar ddof");
}
TEST_CASE("nan-reductions degrade to plain on int (no NaN)") {
  ndarray i = arange(0.0, 6.0, 1.0).astype(kInt64);
  CHECK(nansum(i).item<int64_t>({}) == 15);
}
