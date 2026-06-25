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

// Real-input transforms.
NUMPP_API ndarray rfft(const ndarray& a, std::optional<int64_t> n = std::nullopt,
                       int64_t axis = -1, const std::string& norm = "backward");
NUMPP_API ndarray irfft(const ndarray& a, std::optional<int64_t> n = std::nullopt,
                        int64_t axis = -1, const std::string& norm = "backward");
NUMPP_API ndarray hfft(const ndarray& a, std::optional<int64_t> n = std::nullopt,
                       int64_t axis = -1, const std::string& norm = "backward");
NUMPP_API ndarray ihfft(const ndarray& a, std::optional<int64_t> n = std::nullopt,
                        int64_t axis = -1, const std::string& norm = "backward");

// N-D transforms (apply the 1-D transform along each axis).
NUMPP_API ndarray fftn(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
                       std::optional<std::vector<int64_t>> axes = std::nullopt, const std::string& norm = "backward");
NUMPP_API ndarray ifftn(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
                        std::optional<std::vector<int64_t>> axes = std::nullopt, const std::string& norm = "backward");
NUMPP_API ndarray fft2(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
                       std::optional<std::vector<int64_t>> axes = std::nullopt, const std::string& norm = "backward");
NUMPP_API ndarray ifft2(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
                        std::optional<std::vector<int64_t>> axes = std::nullopt, const std::string& norm = "backward");
NUMPP_API ndarray rfftn(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
                        std::optional<std::vector<int64_t>> axes = std::nullopt, const std::string& norm = "backward");
NUMPP_API ndarray irfftn(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
                         std::optional<std::vector<int64_t>> axes = std::nullopt, const std::string& norm = "backward");
NUMPP_API ndarray rfft2(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
                        std::optional<std::vector<int64_t>> axes = std::nullopt, const std::string& norm = "backward");
NUMPP_API ndarray irfft2(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
                         std::optional<std::vector<int64_t>> axes = std::nullopt, const std::string& norm = "backward");

// Frequency helpers / shifts.
NUMPP_API ndarray fftfreq(int64_t n, double d = 1.0);
NUMPP_API ndarray rfftfreq(int64_t n, double d = 1.0);
NUMPP_API ndarray fftshift(const ndarray& a, std::optional<std::vector<int64_t>> axes = std::nullopt);
NUMPP_API ndarray ifftshift(const ndarray& a, std::optional<std::vector<int64_t>> axes = std::nullopt);

}  // namespace fft
}  // namespace numpp
