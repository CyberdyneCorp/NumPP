#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <map>
#include <string>

using namespace numpp;

TEST_CASE("geterr returns numpy default policies") {
  // Restore a clean default in case a prior test left state changed.
  seterr(std::map<std::string, std::string>{{"all", "warn"}, {"under", "ignore"}});
  auto e = geterr();
  CHECK(e["divide"] == "warn");
  CHECK(e["over"] == "warn");
  CHECK(e["under"] == "ignore");
  CHECK(e["invalid"] == "warn");
}

TEST_CASE("seterr returns previous state and applies new") {
  seterr(std::string("ignore"));
  auto prev = seterr(std::map<std::string, std::string>{{"divide", "raise"}});
  CHECK(prev["divide"] == "ignore");           // previous value reported
  CHECK(geterr()["divide"] == "raise");        // new value applied
  CHECK(geterr()["over"] == "ignore");         // untouched key keeps prior value
  seterr(std::string("ignore"));               // reset
}

TEST_CASE("seterr rejects an invalid policy") {
  CHECK_THROWS_AS(seterr(std::string("nonsense")), value_error);
  seterr(std::string("ignore"));
}

TEST_CASE("errstate scopes the policy and restores it") {
  seterr(std::string("ignore"));
  {
    errstate guard(std::map<std::string, std::string>{{"divide", "raise"}});
    CHECK(geterr()["divide"] == "raise");
  }
  CHECK(geterr()["divide"] == "ignore");  // restored on scope exit
}

TEST_CASE("divide policy raise throws on division by zero") {
  errstate guard(std::map<std::string, std::string>{{"divide", "raise"}});
  ndarray one = full(Shape{4}, 1.0, kFloat64);
  ndarray zero = zeros(Shape{4}, kFloat64);
  CHECK_THROWS_AS(divide(one, zero), floating_point_error);
}

TEST_CASE("invalid policy raise throws on 0/0 and sqrt(-1)") {
  errstate guard(std::map<std::string, std::string>{{"invalid", "raise"}});
  ndarray zero = zeros(Shape{4}, kFloat64);
  CHECK_THROWS_AS(divide(zero, zero), floating_point_error);
  ndarray neg = full(Shape{4}, -1.0, kFloat64);
  CHECK_THROWS_AS(sqrt(neg), floating_point_error);
}

TEST_CASE("ignore policy yields inf matching numpy") {
  errstate guard(std::map<std::string, std::string>{{"all", "ignore"}});
  ndarray one = full(Shape{3}, 1.0, kFloat64);
  ndarray zero = zeros(Shape{3}, kFloat64);
  ndarray got = divide(one, zero);  // [inf, inf, inf], must not throw
  if (!npt::numpy_available()) return;
  auto o = npt::oracle("import numpy as np\n"
                       "with np.errstate(divide='ignore'):\n"
                       "    a = np.ones(3) / np.zeros(3)");
  if (!o) return;
  CHECK(allclose(got, *o, 1e-12, 1e-12, true));  // equal_nan: compares inf positions
}

TEST_CASE("call policy invokes the callback") {
  std::string seen;
  seterrcall([&](const std::string& cond, int) { seen = cond; });
  errstate guard(std::map<std::string, std::string>{{"divide", "call"}});
  ndarray one = full(Shape{2}, 1.0, kFloat64);
  ndarray zero = zeros(Shape{2}, kFloat64);
  divide(one, zero);  // triggers the "divide" condition
  CHECK(seen == "divide");
  seterrcall(nullptr);
  seterr(std::string("ignore"));
}
