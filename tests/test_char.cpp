#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"

#include <string>
#include <vector>

using namespace numpp;
namespace C = numpp::npchar;

namespace {
bool str_eq(const ndarray& a, const std::vector<std::string>& expected) {
  if (a.size() != static_cast<int64_t>(expected.size())) return false;
  for (int64_t i = 0; i < a.size(); ++i)
    if (get_string(a, i) != expected[i]) return false;
  return true;
}
ndarray sample() { return string_array({"Hello", "World ", " foo", "barBAZ"}); }
const char* PY = "np.array(['Hello','World ',' foo','barBAZ'])";
}  // namespace

// ---- string -> string (direct equality on known results) ----
TEST_CASE("char.upper") { CHECK(str_eq(C::upper(sample()), {"HELLO", "WORLD ", " FOO", "BARBAZ"})); }
TEST_CASE("char.lower") { CHECK(str_eq(C::lower(sample()), {"hello", "world ", " foo", "barbaz"})); }
TEST_CASE("char.strip") { CHECK(str_eq(C::strip(sample()), {"Hello", "World", "foo", "barBAZ"})); }
TEST_CASE("char.lstrip") { CHECK(str_eq(C::lstrip(sample()), {"Hello", "World ", "foo", "barBAZ"})); }
TEST_CASE("char.rstrip") { CHECK(str_eq(C::rstrip(sample()), {"Hello", "World", " foo", "barBAZ"})); }
TEST_CASE("char.capitalize") { CHECK(str_eq(C::capitalize(string_array({"hELLo", "wOR"})), {"Hello", "Wor"})); }
TEST_CASE("char.title") { CHECK(str_eq(C::title(string_array({"hello world", "a-b c"})), {"Hello World", "A-B C"})); }
TEST_CASE("char.add") {
  CHECK(str_eq(C::add(string_array({"a", "b", "c"}), string_array({"x", "y", "z"})), {"ax", "by", "cz"}));
}
TEST_CASE("char.multiply") { CHECK(str_eq(C::multiply(string_array({"ab", "c"}), 3), {"ababab", "ccc"})); }
TEST_CASE("char.replace") {
  CHECK(str_eq(C::replace(string_array({"a-b-c", "x-y"}), "-", "__"), {"a__b__c", "x__y"}));
}

// ---- string -> int64 / bool (numpy oracle) ----
TEST_CASE("char.str_len vs numpy") {
  auto o = npt::oracle(std::string("a=np.char.str_len(") + PY + ")");
  if (o) CHECK(allclose(C::str_len(sample()), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.find vs numpy") {
  auto o = npt::oracle(std::string("a=np.char.find(") + PY + ",'o')");
  if (o) CHECK(allclose(C::find(sample(), "o"), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.count vs numpy") {
  auto o = npt::oracle(std::string("a=np.char.count(") + PY + ",'o')");
  if (o) CHECK(allclose(C::count(sample(), "o"), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.startswith vs numpy") {
  auto o = npt::oracle(std::string("a=np.char.startswith(") + PY + ",'H')");
  if (o) CHECK(allclose(C::startswith(sample(), "H"), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.endswith vs numpy") {
  auto o = npt::oracle(std::string("a=np.char.endswith(") + PY + ",'Z')");
  if (o) CHECK(allclose(C::endswith(sample(), "Z"), *o, 1e-9, 1e-12, true));
}
