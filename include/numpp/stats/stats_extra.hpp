#pragma once

#include <optional>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

struct Histogram2DResult {
  ndarray H;        // (nx, ny) counts
  ndarray xedges;   // nx+1
  ndarray yedges;   // ny+1
};
struct HistogramDDResult {
  ndarray H;                     // D-dimensional counts
  std::vector<ndarray> edges;    // per-dimension bin edges
};

// 2-D histogram of paired samples (numpy.histogram2d).
NUMPP_API Histogram2DResult histogram2d(const ndarray& x, const ndarray& y, int64_t bins = 10);
// N-D histogram; `sample` is (N, D) (numpy.histogramdd).
NUMPP_API HistogramDDResult histogramdd(const ndarray& sample, int64_t bins = 10);

// nan-aware quantile (numpy.nanquantile), q in [0, 1], linear interpolation.
NUMPP_API ndarray nanquantile(const ndarray& a, double q, std::optional<int64_t> axis = std::nullopt);

// Weighted covariance (numpy.cov with aweights); each column is an observation.
NUMPP_API ndarray cov_weighted(const ndarray& m, const ndarray& aweights, bool rowvar = true, int64_t ddof = 1);

}  // namespace numpp
