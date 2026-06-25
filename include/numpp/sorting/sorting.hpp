#pragma once

#include <optional>
#include <string>

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

}  // namespace numpp
