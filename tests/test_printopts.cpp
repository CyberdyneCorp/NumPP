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

TEST_CASE("printopts set/get round-trip") {
  PrintOptions saved = get_printoptions();
  PrintOptions o;
  o.precision = 4;
  o.suppress_small = true;
  o.threshold = 50;
  o.edgeitems = 2;
  o.sign = "+";
  set_printoptions(o);
  PrintOptions g = get_printoptions();
  CHECK(g.precision == 4);
  CHECK(g.suppress_small == true);
  CHECK(g.threshold == 50);
  CHECK(g.edgeitems == 2);
  CHECK(g.sign == "+");
  set_printoptions(saved);  // restore
}

TEST_CASE("format_float_positional known cases") {
  CHECK(format_float_positional(1.5) == "1.5");
  CHECK(format_float_positional(1.0) == "1.");
  CHECK(format_float_positional(0.0625, 4) == "0.0625");
  CHECK(format_float_positional(2.0) == "2.");
  CHECK(format_float_positional(1.0, std::nullopt, true, true) == "1");
  CHECK(format_float_positional(123.0, 2) == "123.");
}

TEST_CASE("format_float_scientific known cases") {
  CHECK(format_float_scientific(1234.5, 2) == "1.23e+03");
  CHECK(format_float_scientific(0.0, 1) == "0.0e+00");
}

TEST_CASE("array2string known cases") {
  CHECK(array2string(arange(0., 3., 1., kFloat64)) == "[0. 1. 2.]");
  CHECK((array2string(arange(0., 4., 1., kFloat64).reshape(Shape{2, 2})) == "[[0. 1.]\n [2. 3.]]"));
}
