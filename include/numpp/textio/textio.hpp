#pragma once

#include <cstdint>
#include <string>

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- text I/O ----
// loadtxt: parse a whitespace/delimiter-separated text table into a float64 array
// (1-D when a single row or column, else 2-D). '#' lines are comments.
NUMPP_API ndarray loadtxt(const std::string& path, char delimiter = '\0');
NUMPP_API void savetxt(const std::string& path, const ndarray& a,
                       const std::string& fmt = "%.18e", const std::string& delimiter = " ");
// genfromtxt: like loadtxt; missing/unparsable fields become NaN.
NUMPP_API ndarray genfromtxt(const std::string& path, char delimiter = '\0');
// fromstring: parse a separated list of numbers into a 1-D float64 array.
NUMPP_API ndarray fromstring(const std::string& s, const std::string& sep);

// ---- raw binary I/O ----
NUMPP_API void tofile(const ndarray& a, const std::string& path);
NUMPP_API ndarray fromfile(const std::string& path, DType dtype, int64_t count = -1);

// ---- integer string formatting ----
NUMPP_API std::string binary_repr(int64_t value, int64_t width = 0);
NUMPP_API std::string base_repr(int64_t value, int64_t base = 2, int64_t padding = 0);

}  // namespace numpp
