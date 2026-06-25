#pragma once

#include <optional>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {

// ---- joining ----
NUMPP_API ndarray concatenate(const std::vector<ndarray>& arrays, int64_t axis = 0);
NUMPP_API ndarray stack(const std::vector<ndarray>& arrays, int64_t axis = 0);
NUMPP_API ndarray hstack(const std::vector<ndarray>& arrays);
NUMPP_API ndarray vstack(const std::vector<ndarray>& arrays);
NUMPP_API ndarray dstack(const std::vector<ndarray>& arrays);
NUMPP_API ndarray column_stack(const std::vector<ndarray>& arrays);

// ---- splitting ----
NUMPP_API std::vector<ndarray> array_split(const ndarray& a, int64_t sections, int64_t axis = 0);
NUMPP_API std::vector<ndarray> split(const ndarray& a, int64_t sections, int64_t axis = 0);
NUMPP_API std::vector<ndarray> hsplit(const ndarray& a, int64_t sections);
NUMPP_API std::vector<ndarray> vsplit(const ndarray& a, int64_t sections);

// ---- tiling ----
NUMPP_API ndarray tile(const ndarray& a, const std::vector<int64_t>& reps);
NUMPP_API ndarray repeat(const ndarray& a, int64_t repeats, std::optional<int64_t> axis = std::nullopt);

// ---- rearranging ----
NUMPP_API ndarray flip(const ndarray& a, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray fliplr(const ndarray& a);
NUMPP_API ndarray flipud(const ndarray& a);
NUMPP_API ndarray roll(const ndarray& a, int64_t shift, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray rot90(const ndarray& a, int64_t k = 1);  // in the (0,1) plane
NUMPP_API ndarray moveaxis(const ndarray& a, int64_t source, int64_t destination);
NUMPP_API ndarray atleast_1d(const ndarray& a);
NUMPP_API ndarray atleast_2d(const ndarray& a);
NUMPP_API ndarray atleast_3d(const ndarray& a);

// ---- adding/removing elements ----
NUMPP_API ndarray append(const ndarray& a, const ndarray& values, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray insert(const ndarray& a, int64_t index, const ndarray& values, std::optional<int64_t> axis = std::nullopt);
NUMPP_API ndarray delete_(const ndarray& a, int64_t index, std::optional<int64_t> axis = std::nullopt);  // numpy.delete
NUMPP_API ndarray resize(const ndarray& a, const Shape& new_shape);
// pad mode: "constant" (with constant_value) or "edge".
NUMPP_API ndarray pad(const ndarray& a, const std::vector<std::pair<int64_t, int64_t>>& pad_width,
                      const std::string& mode = "constant", double constant_value = 0.0);

}  // namespace numpp
