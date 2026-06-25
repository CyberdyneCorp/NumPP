#pragma once

#include <optional>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

enum class SortKind { Quicksort, Mergesort, Heapsort, Stable };

// axis default = last axis; std::nullopt = flatten (numpy axis=None).
NUMPP_API ndarray sort(const ndarray& a, std::optional<int64_t> axis = int64_t{-1},
                       SortKind kind = SortKind::Quicksort);
NUMPP_API ndarray argsort(const ndarray& a, std::optional<int64_t> axis = int64_t{-1},
                          SortKind kind = SortKind::Quicksort);

}  // namespace numpp
