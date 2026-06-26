#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace numpp;

TEST_CASE("testing array_equal identical and different") {
  auto x = arange(0., 6., 1., kFloat64);
  auto y = arange(0., 6., 1., kFloat64);
  auto z = arange(1., 7., 1., kFloat64);
  CHECK(testing::array_equal(x, x));
  CHECK(testing::array_equal(x, y));
  CHECK(!testing::array_equal(x, z));
}

TEST_CASE("testing array_equal shape mismatch is false") {
  auto x = arange(0., 6., 1., kFloat64);
  auto y = arange(0., 5., 1., kFloat64);
  CHECK(!testing::array_equal(x, y));
}

TEST_CASE("testing array_equiv broadcasting equal") {
  auto row = arange(0., 3., 1., kFloat64);       // shape (3,)
  auto tiled = full({2, 3}, 0.0, kFloat64);
  // tiled rows are [0,0,0]; not equal to [0,1,2] -> false
  CHECK(!testing::array_equiv(row, tiled));
  // scalar-like (1,) broadcast against (2,3)
  auto s = full({1}, 5.0, kFloat64);
  auto fives = full({2, 3}, 5.0, kFloat64);
  CHECK(testing::array_equiv(s, fives));
}

TEST_CASE("testing array_equiv non-broadcastable returns false") {
  auto a = arange(0., 4., 1., kFloat64);  // (4,)
  auto b = arange(0., 3., 1., kFloat64);  // (3,)
  CHECK(!testing::array_equiv(a, b));
}

TEST_CASE("testing assert_array_equal throws on difference") {
  auto x = arange(0., 6., 1., kFloat64);
  auto z = arange(1., 7., 1., kFloat64);
  CHECK_THROWS_AS(testing::assert_array_equal(x, z), numpp::value_error);
  testing::assert_array_equal(x, x);
  CHECK(true);
}

TEST_CASE("testing assert_allclose passes within tolerance") {
  auto x = arange(0., 6., 1., kFloat64);
  auto xp = arange(0., 6., 1., kFloat64);
  xp.set_item<double>({3}, 3.0 + 1e-10);  // tiny perturbation on a nonzero element (rtol applies)
  testing::assert_allclose(x, xp);
  CHECK(true);
}

TEST_CASE("testing assert_allclose throws when far") {
  auto x = arange(0., 6., 1., kFloat64);
  auto far = arange(10., 16., 1., kFloat64);
  CHECK_THROWS_AS(testing::assert_allclose(x, far), numpp::value_error);
}

TEST_CASE("testing assert_array_almost_equal") {
  auto x = arange(0., 4., 1., kFloat64);
  auto xp = arange(0., 4., 1., kFloat64);
  xp.set_item<double>({1}, 1.0 + 1e-9);
  testing::assert_array_almost_equal(x, xp, 6);
  CHECK(true);
  auto bad = arange(0., 4., 1., kFloat64);
  bad.set_item<double>({1}, 1.1);
  CHECK_THROWS_AS(testing::assert_array_almost_equal(x, bad, 6), numpp::value_error);
}

TEST_CASE("testing assert_array_less") {
  auto a = arange(0., 4., 1., kFloat64);
  auto b = arange(1., 5., 1., kFloat64);
  testing::assert_array_less(a, b);
  CHECK(true);
  CHECK_THROWS_AS(testing::assert_array_less(b, a), numpp::value_error);
  CHECK_THROWS_AS(testing::assert_array_less(a, a), numpp::value_error);
}
