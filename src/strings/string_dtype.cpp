#include "numpp/strings/string_dtype.hpp"

#include "numpp/core/error.hpp"

namespace numpp {
namespace strings {
namespace {

int64_t shape_size(const Shape& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// Pick the element of `o` paired with flat position `i` of the primary operand,
// allowing a single-element operand to broadcast.
const std::string& pair_at(const StringDType& self, const StringDType& o, int64_t i) {
  if (o.size() == self.size()) return o.item(i);
  if (o.size() == 1) return o.item(0);
  throw value_error("StringDType: operands could not be broadcast together");
}

}  // namespace

StringDType::StringDType(std::vector<std::string> data, Shape shape)
    : data_(std::move(data)), shape_(std::move(shape)) {
  if (shape_size(shape_) != static_cast<int64_t>(data_.size()))
    throw value_error("StringDType: data size does not match shape");
}

StringDType StringDType::from_list(const std::vector<std::string>& items) {
  return StringDType(items, Shape{static_cast<int64_t>(items.size())});
}

const std::string& StringDType::item(int64_t flat) const {
  if (flat < 0 || flat >= size()) throw index_error("StringDType: flat index out of range");
  return data_[static_cast<size_t>(flat)];
}

std::string StringDType::item(const std::vector<int64_t>& idx) const {
  if (static_cast<int64_t>(idx.size()) != ndim())
    throw index_error("StringDType: wrong number of indices");
  int64_t flat = 0;
  for (int64_t d = 0; d < ndim(); ++d) {
    int64_t k = idx[static_cast<size_t>(d)];
    if (k < 0 || k >= shape_[static_cast<size_t>(d)]) throw index_error("StringDType: index out of range");
    flat = flat * shape_[static_cast<size_t>(d)] + k;
  }
  return data_[static_cast<size_t>(flat)];
}

void StringDType::set_item(int64_t flat, std::string v) {
  if (flat < 0 || flat >= size()) throw index_error("StringDType: flat index out of range");
  data_[static_cast<size_t>(flat)] = std::move(v);
}

ndarray StringDType::equal(const StringDType& other) const {
  ndarray out(shape_, kBool, Order::C);
  const int64_t n = size();
  bool* o = n ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = (item(i) == pair_at(*this, other, i));
  return out;
}

ndarray StringDType::not_equal(const StringDType& other) const {
  ndarray out(shape_, kBool, Order::C);
  const int64_t n = size();
  bool* o = n ? out.typed_data<bool>() : nullptr;
  for (int64_t i = 0; i < n; ++i) o[i] = (item(i) != pair_at(*this, other, i));
  return out;
}

StringDType StringDType::add(const StringDType& other) const {
  const int64_t n = size();
  std::vector<std::string> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = item(i) + pair_at(*this, other, i);
  return StringDType(std::move(out), shape_);
}

}  // namespace strings
}  // namespace numpp
