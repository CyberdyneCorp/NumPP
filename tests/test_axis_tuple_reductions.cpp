// Axis-tuple reductions (issue #3): reduce over several axes at once, e.g.
// sum(a, {0,2}). Validated against the live-NumPy oracle.
#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <string>
#include <vector>

using namespace numpp;

namespace {
// A 3-D test array with distinct, non-degenerate values.
ndarray A3() { return add(arange(0.0, 24.0, 1.0, kFloat64).reshape({2, 3, 4}), full({2, 3, 4}, 1.0, kFloat64)); }
const char* PYA = "A=(np.arange(24.).reshape(2,3,4)+1); ";
}  // namespace

TEST_CASE("axis-tuple: sum over pairs and triples vs numpy") {
  ndarray A = A3();
  auto chk = [&](const ndarray& got, const std::string& axpy) {
    auto o = npt::oracle(std::string(PYA) + "a=np.sum(A,axis=" + axpy + ")");
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  chk(sum(A, {0, 2}), "(0,2)");
  chk(sum(A, {0, 1}), "(0,1)");
  chk(sum(A, {1, 2}), "(1,2)");
  chk(sum(A, {0, 1, 2}), "(0,1,2)");
  chk(sum(A, {-1, -3}), "(-1,-3)");  // negative axes normalize to (0,2)
}

TEST_CASE("axis-tuple: keepdims re-inserts size-1 dims vs numpy") {
  ndarray A = A3();
  ndarray got = sum(A, {0, 2}, /*keepdims=*/true);
  CHECK((got.shape() == Shape{1, 3, 1}));
  auto o = npt::oracle(std::string(PYA) + "a=np.sum(A,axis=(0,2),keepdims=True)");
  if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
}

TEST_CASE("axis-tuple: prod / mean / amin / amax over axes vs numpy") {
  ndarray A = A3();
  auto chk = [&](const ndarray& got, const std::string& fn, const std::string& axpy) {
    auto o = npt::oracle(std::string(PYA) + "a=np." + fn + "(A,axis=" + axpy + ")");
    if (o) CHECK(allclose(got, *o, 1e-9, 1e-11, true));
  };
  chk(prod(A, {1, 2}), "prod", "(1,2)");
  chk(mean(A, {0, 2}), "mean", "(0,2)");
  chk(amin(A, {0, 1}), "min", "(0,1)");
  chk(amax(A, {1, 2}), "max", "(1,2)");
}

TEST_CASE("axis-tuple: var / std (ddof) over axes vs numpy") {
  ndarray A = A3();
  auto ov = npt::oracle(std::string(PYA) + "a=np.var(A,axis=(0,2))");
  if (ov) CHECK(allclose(var(A, {0, 2}), *ov, 1e-9, 1e-11, true));
  auto ovd = npt::oracle(std::string(PYA) + "a=np.var(A,axis=(0,2),ddof=1)");
  if (ovd) CHECK(allclose(var(A, {0, 2}, false, 1), *ovd, 1e-9, 1e-11, true));
  auto os = npt::oracle(std::string(PYA) + "a=np.std(A,axis=(1,2))");
  if (os) CHECK(allclose(numpp::std(A, {1, 2}), *os, 1e-9, 1e-11, true));
}

TEST_CASE("axis-tuple: any / all over axes vs numpy") {
  // Mixed zero/nonzero pattern so any/all differ per slice.
  ndarray B(Shape{2, 2, 2}, kFloat64, Order::C);
  double v[8] = {0, 1, 0, 0, 1, 1, 0, 1};
  for (int i = 0; i < 8; ++i) B.typed_data<double>()[i] = v[i];
  const char* PYB = "B=np.array([0,1,0,0,1,1,0,1.]).reshape(2,2,2); ";
  auto oa = npt::oracle(std::string(PYB) + "a=np.any(B,axis=(0,2))");
  if (oa) CHECK(allclose(any(B, {0, 2}).astype(kFloat64), *oa, 0, 0, true));
  auto ol = npt::oracle(std::string(PYB) + "a=np.all(B,axis=(0,2))");
  if (ol) CHECK(allclose(all(B, {0, 2}).astype(kFloat64), *ol, 0, 0, true));
}

TEST_CASE("axis-tuple: empty axes reduces nothing; full axes == scalar") {
  ndarray A = A3();
  CHECK((sum(A, std::vector<int64_t>{}).shape() == Shape{2, 3, 4}));  // axis=() -> unchanged shape
  CHECK(sum(A, {0, 1, 2}).item<double>({}) == 300.0);                 // 1+2+...+24
}
