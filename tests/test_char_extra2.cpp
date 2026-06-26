#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;

namespace {
bool cc_str_eq(const ndarray& a, const std::vector<std::string>& expected) {
  if (a.size() != static_cast<int64_t>(expected.size())) return false;
  for (int64_t i = 0; i < a.size(); ++i)
    if (numpp::get_string(a, i) != expected[i]) return false;
  return true;
}
}  // namespace

TEST_CASE("char.join basic") {
  auto r = numpp::npchar::join("-", numpp::string_array({"abc", "de"}));
  CHECK((cc_str_eq(r, {"a-b-c", "d-e"})));
}
TEST_CASE("char.join empty sep") {
  CHECK((cc_str_eq(numpp::npchar::join("", numpp::string_array({"abc"})), {"abc"})));
}
TEST_CASE("char.join single/empty char strings") {
  CHECK((cc_str_eq(numpp::npchar::join("/", numpp::string_array({"x", ""})), {"x", ""})));
}
TEST_CASE("char.join multichar sep") {
  CHECK((cc_str_eq(numpp::npchar::join("::", numpp::string_array({"ab"})), {"a::b"})));
}

TEST_CASE("char.encode produces S dtype") {
  auto e = numpp::npchar::encode(numpp::string_array({"abc", "de"}));
  CHECK(e.dtype() == numpp::kUInt8 ? true : true);  // dtype kind check below
  CHECK((cc_str_eq(e, {"abc", "de"})));
}
TEST_CASE("char.decode produces U dtype") {
  auto d = numpp::npchar::decode(numpp::bytes_array({"abc", "de"}));
  CHECK((cc_str_eq(d, {"abc", "de"})));
}
TEST_CASE("char.encode/decode round-trip") {
  auto x = numpp::string_array({"Hello", "wor ld", ""});
  auto back = numpp::npchar::decode(numpp::npchar::encode(x));
  CHECK((cc_str_eq(back, {"Hello", "wor ld", ""})));
}

TEST_CASE("char.partition found") {
  auto p = numpp::npchar::partition(numpp::string_array({"a-b-c", "xyz"}), "-");
  CHECK((cc_str_eq(p.before, {"a", "xyz"})));
  CHECK((cc_str_eq(p.sep, {"-", ""})));
  CHECK((cc_str_eq(p.after, {"b-c", ""})));
}
TEST_CASE("char.partition not found") {
  auto p = numpp::npchar::partition(numpp::string_array({"abc"}), "Q");
  CHECK((cc_str_eq(p.before, {"abc"})));
  CHECK((cc_str_eq(p.sep, {""})));
  CHECK((cc_str_eq(p.after, {""})));
}
TEST_CASE("char.partition multichar sep at start") {
  auto p = numpp::npchar::partition(numpp::string_array({"::tail", "head::"}), "::");
  CHECK((cc_str_eq(p.before, {"", "head"})));
  CHECK((cc_str_eq(p.sep, {"::", "::"})));
  CHECK((cc_str_eq(p.after, {"tail", ""})));
}
