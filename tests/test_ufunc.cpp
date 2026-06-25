#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

// Helper: compare a NumPP result to a NumPy expression (skips if no oracle).
static void chk(const ndarray& got, const std::string& code, const char* label) {
  auto o = npt::oracle(code);
  if (!o) { std::fprintf(stderr, "  [skip] %s (no oracle)\n", label); return; }
  CHECK(allclose(got, *o, 1e-6, 1e-9, /*equal_nan=*/true));
}

// Reproducible inputs mirrored on both sides.
static ndarray A() { return arange(1.0, 7.0, 1.0, kFloat64).reshape({2, 3}); }   // [[1,2,3],[4,5,6]]
static ndarray B() { return arange(2.0, 8.0, 1.0, kFloat64).reshape({2, 3}); }   // [[2..],[..7]]
static const char* Apy = "A=np.arange(1,7,dtype=np.float64).reshape(2,3)";
static const char* Bpy = "B=np.arange(2,8,dtype=np.float64).reshape(2,3)";

TEST_CASE("arithmetic ufuncs vs numpy") {
  chk(add(A(), B()), std::string(Apy) + ";" + Bpy + ";a=A+B", "add");
  chk(subtract(A(), B()), std::string(Apy) + ";" + Bpy + ";a=A-B", "subtract");
  chk(multiply(A(), B()), std::string(Apy) + ";" + Bpy + ";a=A*B", "multiply");
  chk(divide(A(), B()), std::string(Apy) + ";" + Bpy + ";a=A/B", "divide");
  chk(floor_divide(A(), B()), std::string(Apy) + ";" + Bpy + ";a=A//B", "floor_divide");
  chk(mod(A(), B()), std::string(Apy) + ";" + Bpy + ";a=A%B", "mod");
  chk(power(A(), B()), std::string(Apy) + ";" + Bpy + ";a=A**B", "power");
  chk(minimum(A(), B()), std::string(Apy) + ";" + Bpy + ";a=np.minimum(A,B)", "minimum");
  chk(maximum(A(), B()), std::string(Apy) + ";" + Bpy + ";a=np.maximum(A,B)", "maximum");
}

TEST_CASE("broadcasting and operators") {
  ndarray col = arange(0.0, 3.0, 1.0).reshape({3, 1});
  ndarray row = arange(0.0, 4.0, 1.0).reshape({1, 4});
  chk(col + row, "a=np.arange(3.0).reshape(3,1)+np.arange(4.0).reshape(1,4)", "bcast add");
  CHECK((A() + B()).shape() == Shape({2, 3}));
  CHECK(array_equal(A() * B(), multiply(A(), B())));
}

TEST_CASE("integer true-division promotes to float64") {
  ndarray ai = arange(1.0, 5.0, 1.0, kFloat64).astype(kInt32);
  ndarray bi = arange(2.0, 6.0, 1.0, kFloat64).astype(kInt32);
  ndarray r = divide(ai, bi);
  CHECK(r.dtype() == kFloat64);
  chk(r, "a=np.arange(1,5,dtype=np.int32)/np.arange(2,6,dtype=np.int32)", "int truediv");
}

TEST_CASE("comparison ufuncs return bool") {
  ndarray r = less(A(), B());
  CHECK(r.dtype() == kBool);
  chk(r, std::string(Apy) + ";" + Bpy + ";a=A<B", "less");
  chk(equal(A(), A()), std::string(Apy) + ";a=A==A", "equal");
  chk(greater_equal(B(), A()), std::string(Apy) + ";" + Bpy + ";a=B>=A", "ge");
}

TEST_CASE("logical and bitwise") {
  ndarray x = arange(0.0, 4.0, 1.0, kFloat64).astype(kInt32);  // [0,1,2,3]
  ndarray y = full({4}, 1.0, kFloat64).astype(kInt32);          // [1,1,1,1]
  chk(bitwise_and(x, y), "a=np.arange(4,dtype=np.int32)&np.ones(4,dtype=np.int32)", "and");
  chk(bitwise_or(x, y), "a=np.arange(4,dtype=np.int32)|np.ones(4,dtype=np.int32)", "or");
  chk(bitwise_xor(x, y), "a=np.arange(4,dtype=np.int32)^np.ones(4,dtype=np.int32)", "xor");
  chk(invert(x), "a=~np.arange(4,dtype=np.int32)", "invert");
  ndarray bx = arange(0.0, 4.0, 1.0).astype(kBool);
  chk(logical_not(bx), "a=np.logical_not(np.arange(4).astype(bool))", "lnot");
  chk(logical_and(bx, bx), "a=np.logical_and(np.arange(4).astype(bool),np.arange(4).astype(bool))", "land");
}

TEST_CASE("unary math vs numpy") {
  chk(negative(A()), std::string(Apy) + ";a=-A", "negative");
  chk(absolute(subtract(A(), B())), std::string(Apy) + ";" + Bpy + ";a=np.abs(A-B)", "abs");
  chk(sqrt(A()), std::string(Apy) + ";a=np.sqrt(A)", "sqrt");
  chk(exp(A()), std::string(Apy) + ";a=np.exp(A)", "exp");
  chk(log(A()), std::string(Apy) + ";a=np.log(A)", "log");
  chk(sin(A()), std::string(Apy) + ";a=np.sin(A)", "sin");
  chk(cos(A()), std::string(Apy) + ";a=np.cos(A)", "cos");
  chk(floor(divide(A(), B())), std::string(Apy) + ";" + Bpy + ";a=np.floor(A/B)", "floor");
  chk(square(A()), std::string(Apy) + ";a=np.square(A)", "square");
}

TEST_CASE("sqrt of int promotes to float64") {
  ndarray i = arange(1.0, 5.0, 1.0, kFloat64).astype(kInt64);
  ndarray r = sqrt(i);
  CHECK(r.dtype() == kFloat64);
  chk(r, "a=np.sqrt(np.arange(1,5,dtype=np.int64))", "sqrt int");
}

TEST_CASE("complex abs returns real") {
  ndarray c = arange(1.0, 4.0, 1.0, kFloat64).astype(kComplex128);
  CHECK(absolute(c).dtype() == kFloat64);
}

TEST_CASE("reductions vs numpy") {
  chk(sum(A()), std::string(Apy) + ";a=np.array(np.sum(A))", "sum all");
  chk(sum(A(), 0), std::string(Apy) + ";a=np.sum(A,axis=0)", "sum axis0");
  chk(sum(A(), 1), std::string(Apy) + ";a=np.sum(A,axis=1)", "sum axis1");
  chk(sum(A(), 1, true), std::string(Apy) + ";a=np.sum(A,axis=1,keepdims=True)", "sum keepdims");
  CHECK(sum(A(), 1, true).shape() == Shape({2, 1}));
  chk(prod(A(), 0), std::string(Apy) + ";a=np.prod(A,axis=0)", "prod axis0");
  chk(mean(A(), 1), std::string(Apy) + ";a=np.mean(A,axis=1)", "mean axis1");
  chk(amin(A(), 0), std::string(Apy) + ";a=np.min(A,axis=0)", "min axis0");
  chk(amax(A(), 1), std::string(Apy) + ";a=np.max(A,axis=1)", "max axis1");
}

TEST_CASE("integer sum upcasts to int64") {
  ndarray i8 = arange(1.0, 5.0, 1.0, kFloat64).astype(kInt8);
  ndarray r = sum(i8);
  CHECK(r.dtype() == kInt64);
  chk(r, "a=np.array(np.sum(np.arange(1,5,dtype=np.int8)))", "int8 sum");
}

TEST_CASE("any/all") {
  ndarray z = zeros({2, 3}, kFloat64);
  ndarray m = A();
  CHECK(any(m).item<bool>({}) == true);
  CHECK(all(m).item<bool>({}) == true);
  CHECK(any(z).item<bool>({}) == false);
  chk(any(m, 0), std::string(Apy) + ";a=np.any(A,axis=0)", "any axis0");
}
