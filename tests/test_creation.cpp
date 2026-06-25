#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

using namespace numpp;

TEST_CASE("zeros, ones, full") {
  ndarray z = zeros({2, 2}, kFloat64);
  CHECK(z.item<double>({0, 0}) == 0.0 && z.item<double>({1, 1}) == 0.0);
  ndarray o = ones({2}, kInt32);
  CHECK(o.item<int32_t>({0}) == 1 && o.item<int32_t>({1}) == 1);
  ndarray f = full({3}, 7.0, kFloat64);
  CHECK(f.item<double>({2}) == 7.0);
}

TEST_CASE("like-constructors") {
  ndarray a({2, 3}, kFloat32);
  ndarray z = zeros_like(a);
  CHECK(z.shape() == a.shape() && z.dtype() == kFloat32);
  CHECK(z.item<float>({0, 0}) == 0.0f);
}

TEST_CASE("arange") {
  ndarray a = arange(0.0, 10.0, 2.0, kInt64);
  CHECK(a.size() == 5);
  CHECK(a.item<int64_t>({0}) == 0 && a.item<int64_t>({4}) == 8);
}

TEST_CASE("eye and identity") {
  ndarray e = eye(3, 3, 1, kFloat64);
  CHECK(e.item<double>({0, 1}) == 1.0);
  CHECK(e.item<double>({0, 0}) == 0.0);
  ndarray id = identity(3);
  CHECK(id.item<double>({0, 0}) == 1.0 && id.item<double>({0, 1}) == 0.0);
}

TEST_CASE("linspace endpoints") {
  ndarray l = linspace(0.0, 1.0, 5);
  CHECK(l.item<double>({0}) == 0.0);
  CHECK(l.item<double>({4}) == 1.0);
  CHECK(std::abs(l.item<double>({1}) - 0.25) < 1e-12);
}

TEST_CASE("creation vs NumPy oracle") {
  auto check = [](const ndarray& got, const std::string& code) {
    auto o = npt::oracle(code);
    if (!o) { std::fprintf(stderr, "  [skip] %s (no numpy)\n", code.c_str()); return; }
    CHECK(allclose(got, *o));
  };
  check(arange(0.0, 10.0, 2.0, kFloat64), "a=np.arange(0,10,2,dtype=np.float64)");
  check(linspace(0.0, 1.0, 7), "a=np.linspace(0,1,7)");
  check(eye(4, 4, -1, kFloat64), "a=np.eye(4,4,-1)");
  check(full({2, 3}, 2.5, kFloat64), "a=np.full((2,3),2.5)");
}
