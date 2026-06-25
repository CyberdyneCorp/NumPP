#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s (no oracle)\n", label); return; }
  CHECK(allclose(got, *o, 1e-6, 1e-9, /*equal_nan=*/true));
}

static ndarray A() { return arange(1.0, 7.0, 1.0, kFloat64).reshape({2, 3}); }
static const char* Apy = "A=np.arange(1,7,dtype=np.float64).reshape(2,3)";

TEST_CASE("trig and hyperbolic vs numpy") {
  chk(sin(A()), std::string(Apy) + ";a=np.sin(A)", "sin");
  chk(arctan(A()), std::string(Apy) + ";a=np.arctan(A)", "arctan");
  chk(tanh(A()), std::string(Apy) + ";a=np.tanh(A)", "tanh");
  chk(arcsinh(A()), std::string(Apy) + ";a=np.arcsinh(A)", "arcsinh");
  ndarray u = linspace(-0.9, 0.9, 5);
  chk(arcsin(u), "a=np.arcsin(np.linspace(-0.9,0.9,5))", "arcsin");
  chk(deg2rad(A()), std::string(Apy) + ";a=np.deg2rad(A)", "deg2rad");
}

TEST_CASE("math extras vs numpy") {
  chk(cbrt(A()), std::string(Apy) + ";a=np.cbrt(A)", "cbrt");
  chk(log2(A()), std::string(Apy) + ";a=np.log2(A)", "log2");
  chk(log10(A()), std::string(Apy) + ";a=np.log10(A)", "log10");
  chk(expm1(A()), std::string(Apy) + ";a=np.expm1(A)", "expm1");
  chk(log1p(A()), std::string(Apy) + ";a=np.log1p(A)", "log1p");
  chk(trunc(divide(A(), full({2, 3}, 2.0))), std::string(Apy) + ";a=np.trunc(A/2.0)", "trunc");
  chk(rint(divide(A(), full({2, 3}, 2.0))), std::string(Apy) + ";a=np.rint(A/2.0)", "rint");
}

TEST_CASE("log2 of complex matches numpy") {
  ndarray c = A().astype(kComplex128);
  chk(log2(c), std::string(Apy) + ";a=np.log2(A.astype(np.complex128))", "log2 complex");
}

TEST_CASE("binary float: arctan2, hypot, copysign") {
  ndarray x = arange(1.0, 5.0, 1.0);
  ndarray y = arange(4.0, 0.0, -1.0);
  chk(arctan2(x, y), "a=np.arctan2(np.arange(1,5.),np.arange(4,0,-1.))", "arctan2");
  chk(hypot(x, y), "a=np.hypot(np.arange(1,5.),np.arange(4,0,-1.))", "hypot");
  ndarray s = arange(-2.0, 2.0, 1.0);
  chk(copysign(full({4}, 3.0), s), "a=np.copysign(np.full(4,3.0),np.arange(-2,2,1.0))", "copysign");
}

TEST_CASE("predicates: isnan/isinf/isfinite/signbit") {
  ndarray v({4}, kFloat64);
  v.set_item<double>({0}, std::numeric_limits<double>::quiet_NaN());
  v.set_item<double>({1}, std::numeric_limits<double>::infinity());
  v.set_item<double>({2}, -3.0);
  v.set_item<double>({3}, 5.0);
  const char* vpy = "v=np.array([np.nan,np.inf,-3.0,5.0])";
  chk(isnan(v), std::string(vpy) + ";a=np.isnan(v)", "isnan");
  chk(isinf(v), std::string(vpy) + ";a=np.isinf(v)", "isinf");
  chk(isfinite(v), std::string(vpy) + ";a=np.isfinite(v)", "isfinite");
  chk(signbit(v), std::string(vpy) + ";a=np.signbit(v)", "signbit");
}

TEST_CASE("shifts and fmin/fmax") {
  ndarray x = arange(0.0, 5.0, 1.0).astype(kInt32);
  chk(left_shift(x, full({5}, 1.0).astype(kInt32)),
      "a=np.left_shift(np.arange(5,dtype=np.int32),np.ones(5,dtype=np.int32))", "lshift");
  chk(right_shift(full({5}, 16.0).astype(kInt32), x),
      "a=np.right_shift(np.full(5,16,dtype=np.int32),np.arange(5,dtype=np.int32))", "rshift");
  ndarray p = arange(1.0, 5.0, 1.0), q = arange(4.0, 0.0, -1.0);
  chk(fmin(p, q), "a=np.fmin(np.arange(1,5.),np.arange(4,0,-1.))", "fmin");
}

TEST_CASE("minimum/maximum propagate NaN (regression for issue)") {
  ndarray x({2}, kFloat64), y({2}, kFloat64);
  x.set_item<double>({0}, std::numeric_limits<double>::quiet_NaN());
  x.set_item<double>({1}, 5.0);
  y.set_item<double>({0}, 1.0);
  y.set_item<double>({1}, 2.0);
  const char* code = "x=np.array([np.nan,5.0]);y=np.array([1.0,2.0])";
  chk(minimum(x, y), std::string(code) + ";a=np.minimum(x,y)", "minimum nan");
  chk(maximum(x, y), std::string(code) + ";a=np.maximum(x,y)", "maximum nan");
  chk(fmin(x, y), std::string(code) + ";a=np.fmin(x,y)", "fmin nan");  // ignores nan
}

TEST_CASE("clip, var, std vs numpy") {
  chk(clip(A(), full({2, 3}, 2.0), full({2, 3}, 5.0)), std::string(Apy) + ";a=np.clip(A,2.0,5.0)", "clip");
  chk(var(A()), std::string(Apy) + ";a=np.array(np.var(A))", "var all");
  chk(var(A(), 1), std::string(Apy) + ";a=np.var(A,axis=1)", "var axis1");
  chk(numpp::std(A(), 0), std::string(Apy) + ";a=np.std(A,axis=0)", "std axis0");
  chk(var(A(), 1, false, 1), std::string(Apy) + ";a=np.var(A,axis=1,ddof=1)", "var ddof1");
}

TEST_CASE("where vs numpy") {
  ndarray cond = greater(A(), full({2, 3}, 3.0));
  chk(where(cond, A(), negative(A())), std::string(Apy) + ";a=np.where(A>3.0,A,-A)", "where");
}

TEST_CASE("nonzero vs numpy") {
  ndarray m({2, 3}, kFloat64);
  m.fill<double>(0.0);
  m.set_item<double>({0, 1}, 7.0);
  m.set_item<double>({1, 2}, 9.0);
  auto nz = nonzero(m);
  auto o0 = npt::oracle("m=np.zeros((2,3));m[0,1]=7;m[1,2]=9;a=np.nonzero(m)[0]");
  auto o1 = npt::oracle("m=np.zeros((2,3));m[0,1]=7;m[1,2]=9;a=np.nonzero(m)[1]");
  if (!o0 || !o1) { std::fprintf(stderr, "  [skip] nonzero (no oracle)\n"); return; }
  CHECK(nz.size() == 2);
  CHECK(array_equal(nz[0], *o0));
  CHECK(array_equal(nz[1], *o1));
}

TEST_CASE("comparison operators and scalar operands") {
  // weak promotion: int array + int scalar stays int; + float scalar -> float64
  ndarray i = arange(0.0, 4.0, 1.0).astype(kInt32);
  CHECK((i + 1ll).dtype() == kInt32);
  CHECK((i + 1.0).dtype() == kFloat64);
  chk(i + 1ll, "a=np.arange(4,dtype=np.int32)+1", "int+int scalar");
  chk(i * 2.0, "a=np.arange(4,dtype=np.int32)*2.0", "int*float scalar");
  chk((A() > 3.0), std::string(Apy) + ";a=A>3.0", "gt scalar");
  CHECK((A() < A()).dtype() == kBool);
  CHECK(array_equal(A() + A(), multiply(A(), full({2, 3}, 2.0))));
}
