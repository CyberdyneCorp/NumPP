#pragma once

#include <optional>
#include <string>
#include <utility>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

enum class SortKind { Quicksort, Mergesort, Heapsort, Stable };

// axis default = last axis; std::nullopt = flatten (numpy axis=None).
NUMPP_API ndarray sort(const ndarray& a, std::optional<int64_t> axis = int64_t{-1},
                       SortKind kind = SortKind::Quicksort);
NUMPP_API ndarray argsort(const ndarray& a, std::optional<int64_t> axis = int64_t{-1},
                          SortKind kind = SortKind::Quicksort);

// Search / select.
NUMPP_API ndarray searchsorted(const ndarray& a, const ndarray& v, const std::string& side = "left");
NUMPP_API ndarray partition(const ndarray& a, int64_t kth, std::optional<int64_t> axis = int64_t{-1});
NUMPP_API ndarray argpartition(const ndarray& a, int64_t kth, std::optional<int64_t> axis = int64_t{-1});
NUMPP_API ndarray argmin(const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray argmax(const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray flatnonzero(const ndarray& a);
NUMPP_API ndarray count_nonzero(const ndarray& a, std::optional<int64_t> axis = std::nullopt);

// Unique / set operations / counts.
struct UniqueResult { ndarray values; ndarray index; ndarray inverse; ndarray counts; };
NUMPP_API ndarray unique(const ndarray& a);
NUMPP_API UniqueResult unique_ex(const ndarray& a, bool return_index, bool return_inverse, bool return_counts);
NUMPP_API ndarray in1d(const ndarray& a, const ndarray& b);
NUMPP_API ndarray isin(const ndarray& a, const ndarray& b);
NUMPP_API ndarray intersect1d(const ndarray& a, const ndarray& b);
NUMPP_API ndarray union1d(const ndarray& a, const ndarray& b);
NUMPP_API ndarray setdiff1d(const ndarray& a, const ndarray& b);
NUMPP_API ndarray bincount(const ndarray& x, const ndarray* weights = nullptr, int64_t minlength = 0);

struct Histogram { ndarray hist; ndarray bin_edges; };
NUMPP_API Histogram histogram(const ndarray& a, int64_t bins = 10,
                              std::optional<std::pair<double, double>> range = std::nullopt);
NUMPP_API ndarray histogram_bin_edges(const ndarray& a, int64_t bins = 10,
                                      std::optional<std::pair<double, double>> range = std::nullopt);

}  // namespace numpp
