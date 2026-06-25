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

TEST_CASE("fft power-of-two vs numpy") {
  chk(fft::fft(ramp(8)), "a=np.fft.fft(np.arange(8.0))", "fft8");
  CHECK(fft::fft(ramp(8)).dtype() == kComplex128);
}

TEST_CASE("fft composite and prime lengths vs numpy") {
  chk(fft::fft(ramp(6)), "a=np.fft.fft(np.arange(6.0))", "fft6");
  chk(fft::fft(ramp(7)), "a=np.fft.fft(np.arange(7.0))", "fft7 prime");
  chk(fft::fft(ramp(12)), "a=np.fft.fft(np.arange(12.0))", "fft12");
  chk(fft::fft(ramp(11)), "a=np.fft.fft(np.arange(11.0))", "fft11 prime");
}

TEST_CASE("ifft round trip and vs numpy") {
  ndarray x = ramp(10);
  CHECK(allclose(fft::ifft(fft::fft(x)), x.astype(kComplex128), 1e-9, 1e-9));
  chk(fft::ifft(ramp(7)), "a=np.fft.ifft(np.arange(7.0))", "ifft7");
}

TEST_CASE("fft of complex input") {
  ndarray c({4}, kComplex128);
  for (int i = 0; i < 4; ++i) c.set_item<std::complex<double>>({i}, {(double)i, (double)(3 - i)});
  chk(fft::fft(c), "a=np.fft.fft(np.array([0+3j,1+2j,2+1j,3+0j]))", "fft complex");
}

TEST_CASE("n parameter: pad and truncate") {
  chk(fft::fft(ramp(5), 8), "a=np.fft.fft(np.arange(5.0),8)", "fft pad");
  chk(fft::fft(ramp(8), 5), "a=np.fft.fft(np.arange(8.0),5)", "fft trunc");
}

TEST_CASE("norm modes vs numpy") {
  chk(fft::fft(ramp(6), std::nullopt, -1, "ortho"), "a=np.fft.fft(np.arange(6.0),norm='ortho')", "ortho");
  chk(fft::fft(ramp(6), std::nullopt, -1, "forward"), "a=np.fft.fft(np.arange(6.0),norm='forward')", "forward");
  chk(fft::ifft(ramp(6), std::nullopt, -1, "ortho"), "a=np.fft.ifft(np.arange(6.0),norm='ortho')", "iortho");
}

TEST_CASE("fft along axis of 2-D array") {
  ndarray a = arange(0.0, 12.0, 1.0).reshape({3, 4});
  chk(fft::fft(a, std::nullopt, 0), "a=np.fft.fft(np.arange(12.0).reshape(3,4),axis=0)", "axis0");
  chk(fft::fft(a, std::nullopt, 1), "a=np.fft.fft(np.arange(12.0).reshape(3,4),axis=1)", "axis1");
}
