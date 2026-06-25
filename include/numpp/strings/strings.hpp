#pragma once

#include <optional>
#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// Create a fixed-width unicode ('U') or bytes ('S') array from C++ strings.
// num_chars/num_bytes default to the longest element.
NUMPP_API ndarray string_array(const std::vector<std::string>& strs,
                               std::optional<int64_t> num_chars = std::nullopt);
NUMPP_API ndarray bytes_array(const std::vector<std::string>& strs,
                              std::optional<int64_t> num_bytes = std::nullopt);

// Element access (1-D, contiguous): UTF-8 std::string view of element i.
NUMPP_API std::string get_string(const ndarray& a, int64_t i);
NUMPP_API void set_string(ndarray& a, int64_t i, const std::string& s);

// Element-wise string comparison -> bool array.
NUMPP_API ndarray str_equal(const ndarray& a, const ndarray& b);
NUMPP_API ndarray str_not_equal(const ndarray& a, const ndarray& b);

}  // namespace numpp
