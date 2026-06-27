#pragma once

#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "numpp/export.hpp"

namespace numpp {
namespace strings {

// Variable-length UTF-8 string array (numpy 2.0 StringDType, dtype "T"). Unlike
// the fixed-width 'U'/'S' dtypes, elements are stored individually and are not
// truncated to a common width. Elements are kept in C (row-major) order.
class NUMPP_API StringDType {
 public:
  StringDType() = default;
  StringDType(std::vector<std::string> data, Shape shape);
  // 1-D array from a list of strings.
  static StringDType from_list(const std::vector<std::string>& items);

  const Shape& shape() const { return shape_; }
  int64_t size() const { return static_cast<int64_t>(data_.size()); }
  int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }

  const std::string& item(int64_t flat) const;
  std::string item(const std::vector<int64_t>& idx) const;
  void set_item(int64_t flat, std::string v);
  const std::vector<std::string>& tolist() const { return data_; }

  // Elementwise comparison (returns a bool ndarray) and concatenation (returns a
  // StringDType). The other operand must match this shape, or be a single
  // element that broadcasts.
  ndarray equal(const StringDType& other) const;
  ndarray not_equal(const StringDType& other) const;
  StringDType add(const StringDType& other) const;

 private:
  std::vector<std::string> data_;  // C-order
  Shape shape_;
};

}  // namespace strings
}  // namespace numpp
