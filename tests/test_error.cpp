#include <exception>

#include "numpp/numpp.hpp"
#include "numpp_test.hpp"

using namespace numpp;

TEST_CASE("errors derive from common base and std::exception") {
  try {
    throw value_error("boom");
  } catch (const error& e) {
    CHECK(std::string(e.what()) == "boom");
    CHECK(dynamic_cast<const std::exception*>(&e) != nullptr);
  }
}

TEST_CASE("typed exceptions thrown by operations") {
  CHECK_THROWS_AS(broadcast_shapes({3}, {4}), value_error);
  CHECK_THROWS_AS(zeros({3}).item<double>({9}), index_error);
  CHECK_THROWS_AS(normalize_axis(5, 2), axis_error);
  CHECK_THROWS_AS(DType::from_name("nope"), type_error);
}

TEST_CASE("strong guarantee: failed reshape leaves input intact") {
  ndarray a = zeros({3, 4}, kFloat64);
  Shape before = a.shape();
  CHECK_THROWS_AS(a.reshape({5, 5}), value_error);
  CHECK(a.shape() == before);
  CHECK(a.c_contiguous());
}

TEST_CASE("non-finite propagates, not throws") {
  ndarray a({2}, kFloat64);
  double inf = std::numeric_limits<double>::infinity();
  a.set_item<double>({0}, inf);
  a.set_item<double>({1}, -inf);
  ndarray b = a.astype(kFloat64);  // element op; must not throw
  CHECK(std::isinf(b.item<double>({0})));
}
