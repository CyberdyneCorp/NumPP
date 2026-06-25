#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s (no oracle)\n", label); return; }
  CHECK(allclose(got, *o, 1e-6, 1e-8, true));
}
static ndarray grid() { return arange(0.0, 24.0, 1.0, kFloat64).reshape({4, 6}); }
static const char* Gpy = "G=np.arange(24.0).reshape(4,6)";

TEST_CASE("fft2 / ifft2 vs numpy and round trip") {
  chk(fft::fft2(grid()), std::string(Gpy) + ";a=np.fft.fft2(G)", "fft2");
  CHECK(allclose(fft::ifft2(fft::fft2(grid())), grid().astype(kComplex128), 1e-9, 1e-9));
}

TEST_CASE("fftn / ifftn 3-D vs numpy") {
  ndarray a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4});
  chk(fft::fftn(a), "a=np.fft.fftn(np.arange(24.0).reshape(2,3,4))", "fftn3d");
  CHECK(allclose(fft::ifftn(fft::fftn(a)), a.astype(kComplex128), 1e-9, 1e-9));
}

TEST_CASE("fft2 with axes and s") {
  chk(fft::fft2(grid(), std::vector<int64_t>{4, 8}), std::string(Gpy) + ";a=np.fft.fft2(G,s=(4,8))", "fft2 s");
  chk(fft::fftn(grid(), std::nullopt, std::vector<int64_t>{0}),
      std::string(Gpy) + ";a=np.fft.fftn(G,axes=(0,))", "fftn axis0");
}

TEST_CASE("rfft2 / irfft2 vs numpy and round trip") {
  ndarray r = fft::rfft2(grid());
  CHECK(r.shape() == Shape({4, 4}));   // last axis 6 -> 6//2+1 = 4
  chk(r, std::string(Gpy) + ";a=np.fft.rfft2(G)", "rfft2");
  CHECK(allclose(fft::irfft2(fft::rfft2(grid()), std::vector<int64_t>{4, 6}), grid(), 1e-9, 1e-9));
}

TEST_CASE("rfftn / irfftn round trip 3-D") {
  ndarray a = arange(0.0, 24.0, 1.0).reshape({2, 3, 4});
  ndarray rt = fft::irfftn(fft::rfftn(a), std::vector<int64_t>{2, 3, 4});
  CHECK(allclose(rt, a, 1e-9, 1e-9));
}
