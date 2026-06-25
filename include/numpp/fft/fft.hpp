#pragma once

#include <optional>
#include <string>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace fft {

// norm: "backward" (default), "ortho", or "forward".
NUMPP_API ndarray fft(const ndarray& a, std::optional<int64_t> n = std::nullopt,
                      int64_t axis = -1, const std::string& norm = "backward");
NUMPP_API ndarray ifft(const ndarray& a, std::optional<int64_t> n = std::nullopt,
                       int64_t axis = -1, const std::string& norm = "backward");

}  // namespace fft
}  // namespace numpp
