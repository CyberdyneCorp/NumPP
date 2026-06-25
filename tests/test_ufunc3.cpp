#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s (no oracle)\n", label); return; }
  CHECK(allclose(got, *o, 1e-6, 1e-9, true));
}

static ndarray cplx() {  // [1+2j, -3+4j, 5-6j]
  ndarray c({3}, kComplex128);
  c.set_item<std::complex<double>>({0}, {1, 2});
  c.set_item<std::complex<double>>({1}, {-3, 4});
  c.set_item<std::complex<double>>({2}, {5, -6});
  return c;
}
static const char* Cpy = "c=np.array([1+2j,-3+4j,5-6j])";

TEST_CASE("conj/real/imag/angle vs numpy") {
  chk(conj(cplx()), std::string(Cpy) + ";a=np.conj(c)", "conj");
  chk(real(cplx()), std::string(Cpy) + ";a=np.real(c)", "real");
  chk(imag(cplx()), std::string(Cpy) + ";a=np.imag(c)", "imag");
  chk(angle(cplx()), std::string(Cpy) + ";a=np.angle(c)", "angle");
  CHECK(real(cplx()).dtype() == kFloat64);
  // real input: real() is identity, imag() is zeros
  ndarray r = arange(1.0, 4.0, 1.0);
  CHECK(array_equal(real(r), r));
  CHECK(array_equal(imag(r), zeros({3}, kFloat64)));
}

TEST_CASE("integer power with negative exponent raises (numpy parity)") {
  ndarray base = arange(1.0, 4.0, 1.0).astype(kInt64);
  ndarray neg = full({3}, -1.0).astype(kInt64);
  CHECK_THROWS_AS(power(base, neg), value_error);
  // float base with negative exponent is fine
  chk(power(arange(1.0, 4.0, 1.0), full({3}, -1.0)),
      "a=np.power(np.arange(1,4.),np.full(3,-1.0))", "float negpow");
}

TEST_CASE("copyto and out= kwargs") {
  ndarray dst = zeros({2, 3}, kFloat64);
  ndarray a = arange(1.0, 7.0, 1.0).reshape({2, 3});
  copyto(dst, a);
  CHECK(array_equal(dst, a));

  // out= writes into and returns the out array
  ndarray out = zeros({2, 3}, kFloat64);
  ndarray r = add(a, a, out);
  CHECK(r.buffer().get() == out.buffer().get());
  chk(out, "a=np.zeros((2,3));x=np.arange(1,7.).reshape(2,3);np.add(x,x,out=a)", "out= add");

  // out= with dtype cast (float result -> int out, like numpy unsafe out)
  ndarray iout = zeros({3}, kFloat64).astype(kInt32);
  multiply(arange(1.0, 4.0, 1.0), full({3}, 2.0), iout);
  chk(iout, "a=np.zeros(3,dtype=np.int32);np.multiply(np.arange(1,4.),2.0,out=a,casting='unsafe')", "out= cast");
}

TEST_CASE("where= masked assignment") {
  ndarray out = full({4}, -1.0, kFloat64);
  ndarray a = arange(0.0, 4.0, 1.0);
  ndarray b = full({4}, 10.0, kFloat64);
  ndarray mask = greater(a, full({4}, 1.0));  // [F,F,T,T]
  add(a, b, out, &mask);                       // only positions 2,3 written
  // expect [-1,-1, 12, 13]
  chk(out, "out=np.full(4,-1.0);a=np.arange(4.0);b=np.full(4,10.0);"
           "np.add(a,b,out=out,where=a>1.0);a=out", "where= add");
}
