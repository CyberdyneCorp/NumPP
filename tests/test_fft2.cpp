#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s (no oracle)\n", label); return; }
  CHECK(allclose(got, *o, 1e-6, 1e-8, true));
}
static ndarray ramp(int64_t n) { return arange(0.0, (double)n, 1.0, kFloat64); }

TEST_CASE("rfft length and values vs numpy") {
  ndarray r = fft::rfft(ramp(8));
  CHECK(r.shape() == Shape({5}));               // 8//2 + 1
  chk(r, "a=np.fft.rfft(np.arange(8.0))", "rfft8");
  chk(fft::rfft(ramp(7)), "a=np.fft.rfft(np.arange(7.0))", "rfft7");
}

TEST_CASE("irfft inverts rfft and matches numpy") {
  ndarray x = ramp(10);
  ndarray rt = fft::irfft(fft::rfft(x), 10);
  CHECK(allclose(rt, x, 1e-9, 1e-9));
  CHECK(rt.dtype() == kFloat64);
  chk(fft::irfft(fft::rfft(ramp(9)), 9), "a=np.fft.irfft(np.fft.rfft(np.arange(9.0)),9)", "irfft9");
}

TEST_CASE("hfft / ihfft vs numpy") {
  ndarray c({4}, kComplex128);
  for (int i = 0; i < 4; ++i) c.set_item<std::complex<double>>({i}, {(double)(i + 1), (double)i});
  chk(fft::hfft(c), "a=np.fft.hfft(np.array([1+0j,2+1j,3+2j,4+3j]))", "hfft");
  chk(fft::ihfft(ramp(6)), "a=np.fft.ihfft(np.arange(6.0))", "ihfft");
}

TEST_CASE("rfft norm modes") {
  chk(fft::rfft(ramp(8), std::nullopt, -1, "ortho"), "a=np.fft.rfft(np.arange(8.0),norm='ortho')", "rfft ortho");
}

TEST_CASE("fftfreq / rfftfreq vs numpy") {
  chk(fft::fftfreq(8, 0.5), "a=np.fft.fftfreq(8,0.5)", "fftfreq even");
  chk(fft::fftfreq(7, 1.0), "a=np.fft.fftfreq(7)", "fftfreq odd");
  chk(fft::rfftfreq(8, 0.5), "a=np.fft.rfftfreq(8,0.5)", "rfftfreq");
}

TEST_CASE("fftshift / ifftshift vs numpy and round trip") {
  chk(fft::fftshift(ramp(6)), "a=np.fft.fftshift(np.arange(6.0))", "fftshift even");
  chk(fft::fftshift(ramp(5)), "a=np.fft.fftshift(np.arange(5.0))", "fftshift odd");
  ndarray x = arange(0.0, 12.0, 1.0).reshape({3, 4});
  CHECK(array_equal(fft::ifftshift(fft::fftshift(x)), x));
  chk(fft::fftshift(x), "a=np.fft.fftshift(np.arange(12.0).reshape(3,4))", "fftshift 2d");
}
