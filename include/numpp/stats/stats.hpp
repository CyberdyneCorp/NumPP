#pragma once

#include <optional>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- cumulative ----
NUMPP_API ndarray cumsum(const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray cumprod(const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray nancumsum(const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray nancumprod(const ndarray& a, std::optional<int64_t> axis = std::nullopt);

// ---- differences ----
NUMPP_API ndarray diff(const ndarray& a, int64_t n = 1, int64_t axis = -1);
NUMPP_API ndarray ediff1d(const ndarray& a);
NUMPP_API ndarray gradient(const ndarray& a);  // 1-D, unit spacing
NUMPP_API ndarray ptp(const ndarray& a, std::optional<int64_t> axis = std::nullopt);

// ---- order statistics (linear interpolation, like numpy default) ----
NUMPP_API ndarray median(const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray percentile(const ndarray& a, double q, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray quantile(const ndarray& a, double q, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray nanmedian(const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray nanpercentile(const ndarray& a, double q, std::optional<int64_t> axis = std::nullopt);

// ---- weighted / correlation / binning ----
NUMPP_API ndarray average(const ndarray& a, std::optional<int64_t> axis = std::nullopt,
                          const ndarray* weights = nullptr);
NUMPP_API ndarray cov(const ndarray& m, bool rowvar = true, int64_t ddof = 1);
NUMPP_API ndarray corrcoef(const ndarray& m, bool rowvar = true);
NUMPP_API ndarray digitize(const ndarray& x, const ndarray& bins, bool right = false);

// ---- nan-aware argmin/argmax ----
NUMPP_API ndarray nanargmin(const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray nanargmax(const ndarray& a, std::optional<int64_t> axis = std::nullopt);

}  // namespace numpp
