#include "numpp/numpp.hpp"
#include "numpp_test.hpp"
#include "oracle.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace numpp;

namespace C = numpp::npchar;

namespace {
bool ce_str_eq(const ndarray& a, const std::vector<std::string>& expected) {
  if (a.size() != static_cast<int64_t>(expected.size())) return false;
  for (int64_t i = 0; i < a.size(); ++i)
    if (get_string(a, i) != expected[i]) return false;
  return true;
}
}  // namespace

// ---- string -> string (direct equality) ----
TEST_CASE("char.center extra") {
  CHECK(ce_str_eq(C::center(string_array({"ab", "abc", "abcd"}), 5),
                  {" ab  ", " abc ", "abcd "}));
}
TEST_CASE("char.center fillchar") {
  CHECK((ce_str_eq(C::center(string_array({"a"}), 5, '*'), {"**a**"})));
}
TEST_CASE("char.ljust") {
  CHECK((ce_str_eq(C::ljust(string_array({"ab", "abcdef"}), 4, '.'), {"ab..", "abcdef"})));
}
TEST_CASE("char.rjust") {
  CHECK((ce_str_eq(C::rjust(string_array({"ab", "abcdef"}), 4, '.'), {"..ab", "abcdef"})));
}
TEST_CASE("char.zfill") {
  CHECK(ce_str_eq(C::zfill(string_array({"42", "-3", "+5", "abcdef"}), 5),
                  {"00042", "-0003", "+0005", "abcdef"}));
}
TEST_CASE("char.swapcase") {
  CHECK(ce_str_eq(C::swapcase(string_array({"Hello", "World123", "AbC"})),
                  {"hELLO", "wORLD123", "aBc"}));
}
TEST_CASE("char.expandtabs") {
  CHECK(ce_str_eq(C::expandtabs(string_array({"a\tb", "ab\tc"}), 4),
                  {"a   b", "ab  c"}));
}

// ---- string -> bool (numpy oracle) ----
TEST_CASE("char.isalpha vs numpy") {
  auto o = npt::oracle("a=np.char.isalpha(np.array(['abc','a1','',' ']))");
  if (o) CHECK(allclose(C::isalpha(string_array({"abc", "a1", "", " "})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.isdigit vs numpy") {
  auto o = npt::oracle("a=np.char.isdigit(np.array(['123','12a','','9']))");
  if (o) CHECK(allclose(C::isdigit(string_array({"123", "12a", "", "9"})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.isspace vs numpy") {
  auto o = npt::oracle("a=np.char.isspace(np.array(['  ',' a','','\\t']))");
  if (o) CHECK(allclose(C::isspace(string_array({"  ", " a", "", "\t"})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.isupper vs numpy") {
  auto o = npt::oracle("a=np.char.isupper(np.array(['ABC','AbC','123','A1']))");
  if (o) CHECK(allclose(C::isupper(string_array({"ABC", "AbC", "123", "A1"})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.islower vs numpy") {
  auto o = npt::oracle("a=np.char.islower(np.array(['abc','aBc','123','a1']))");
  if (o) CHECK(allclose(C::islower(string_array({"abc", "aBc", "123", "a1"})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.isalnum vs numpy") {
  auto o = npt::oracle("a=np.char.isalnum(np.array(['abc123','ab c','','a1']))");
  if (o) CHECK(allclose(C::isalnum(string_array({"abc123", "ab c", "", "a1"})), *o, 1e-9, 1e-12, true));
}
TEST_CASE("char.istitle vs numpy") {
  auto o = npt::oracle("a=np.char.istitle(np.array(['Hello World','Hello world','HELLO','abc']))");
  if (o) CHECK(allclose(C::istitle(string_array({"Hello World", "Hello world", "HELLO", "abc"})), *o, 1e-9, 1e-12, true));
}
