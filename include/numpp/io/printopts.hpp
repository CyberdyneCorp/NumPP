#pragma once

#include <optional>
#include <string>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

struct PrintOptions {
  int precision = 8;
  bool suppress_small = false;   // suppress tiny values as 0
  int64_t threshold = 1000;      // summarize above this many elements
  int64_t edgeitems = 3;
  std::string sign = "-";        // "-", "+", or " "
};

NUMPP_API void set_printoptions(const PrintOptions& opts);
NUMPP_API PrintOptions get_printoptions();

// Format a single float like numpy (numpy.format_float_positional / _scientific).
NUMPP_API std::string format_float_positional(double x, std::optional<int> precision = std::nullopt,
                                              bool unique = true, bool trim_dot = false);
NUMPP_API std::string format_float_scientific(double x, std::optional<int> precision = std::nullopt);

// numpy.array2string honoring precision / suppress_small (1-D and 2-D numeric).
NUMPP_API std::string array2string(const ndarray& a, std::optional<int> precision = std::nullopt,
                                   bool suppress_small = false);

}  // namespace numpp
